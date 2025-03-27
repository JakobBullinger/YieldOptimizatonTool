import streamlit as st
import pandas as pd
import pdfplumber
import re
from datetime import datetime, time
import io

################################################################################
# 1) Session-State: Wir wollen "merged_df" zwischen Button-Klicks speichern
################################################################################
if "merged_df" not in st.session_state:
    st.session_state["merged_df"] = None

################################################################################
# 2) Hilfsfunktionen: Synonym, CSV-Funktionen, PDF-Parsing
################################################################################

### Feisto: MicroTec Übersetzung
synonyms = {
    "38x80": "80x80",
    "17x75": "17x78",
    "17x98": "17x100",
    "17x95": "17x100",
    "23x100": "23x103",
    "47x220": "47x223"
}

def unify_dimension(dim_str):
    return synonyms.get(dim_str, dim_str)

def normalize_dimension(dim_str):
    if pd.isna(dim_str):
        return "Unknown"
    dim_str = (
        str(dim_str)
        .lower()
        .replace('mm', '')
        .replace(',', '.')
        .replace(' ', '')
        .replace('.0', '')
    )
    return dim_str

def parse_vol_eingang(vol_str):
    if pd.isna(vol_str) or vol_str == "":
        return 0.0
    cleaned = str(vol_str).replace('.', '').replace(',', '.')
    try:
        return float(cleaned)
    except:
        return 0.0

def to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Auswertung')
    return output.getvalue()

# CSV-Funktionen
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, header=None, sep=';')
    df.rename(columns={
        1: "Year",
        2: "Month",
        3: "Day",
        4: "Hour",
        5: "Minute",
        15: "Dimension",
        21: "Classification",
        26: "CBM"
    }, inplace=True)
    df['Datetime'] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute']])
    df['Dimension'] = df['Dimension'].apply(normalize_dimension).apply(unify_dimension)
    df['CBM'] = pd.to_numeric(df['CBM'], errors='coerce')
    return df

def filter_data_for_order(df, start_dt, end_dt, dimensions):
    mask = (
        (df['Datetime'] >= start_dt) &
        (df['Datetime'] < end_dt) &
        (df['Dimension'].isin(dimensions))
    )
    return df.loc[mask].copy()

################################################################################
# Neue Funktion: Summiere CBM je Dimension und Klassifikation
################################################################################
def summarize_cbm_by_classifications(df):
    """
    Summiert pro 'Dimension' das Gesamtvolumen (total_cbm) sowie die Teilvolumina
    für jede gewünschte Klassifizierung. Zusätzlich wird der Anteil für 'Waste'
    als Prozentwert (waste_percent) berechnet.
    """
    CLASSIFICATION_MAP = {
        "Waste": "waste_cbm",
        "CE": "ce_cbm",
        "KH I-III": "kh_i_iii_cbm",
        "SF I-III": "sf_i_iii_cbm",
        "SF I-IV": "sf_i_iiii_cbm",
        "SI 0-IV": "si_0_iv_cbm",
        "SI I-II": "si_i_ii_cbm",
        "IND II-III": "ind_ii_iii_cbm",
        "NSI I-III": "nsi_i_iii_cbm",
        "ASS IV": "ass_iv_cbm",
    }

    grouped = df.groupby('Dimension', dropna=False).agg(
        total_cbm=('CBM', 'sum')
    ).reset_index()

    for class_label, col_name in CLASSIFICATION_MAP.items():
        grouped[col_name] = grouped['Dimension'].apply(
            lambda dim: df.loc[
                (df['Dimension'] == dim) &
                (df['Classification'].str.lower() == class_label.lower()),
                'CBM'
            ].sum()
        )

    grouped['waste_percent'] = grouped.apply(
        lambda row: 100 * row['waste_cbm'] / row['total_cbm'] if row['total_cbm'] else 0,
        axis=1
    )

    grouped['total_cbm'] = grouped['total_cbm'].round(3)
    for col_name in CLASSIFICATION_MAP.values():
        grouped[col_name] = grouped[col_name].round(3)
    grouped['waste_percent'] = grouped['waste_percent'].round(2)

    return grouped

# PDF-Parsing
def extract_table_with_suborders_clean(file_input, start_keyword="Auftrag"):
    with pdfplumber.open(file_input) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    
    start_index = full_text.find(start_keyword)
    if start_index == -1:
        st.error(f"Startkeyword '{start_keyword}' nicht gefunden!")
        return None
    table_text = full_text[start_index:]
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]

    main_row_pattern = re.compile(
        r"^(?P<auftrag>\d{5}\s*-\s*.*?)(?=\s+[\d.,]+\s+)"
        r"\s+(?P<stämme>[\d.,]+)\s+"
        r"(?P<vol_eingang>[\d.,]+)\s+"
        r"(?P<durchschn_stammlänge>[\d.,]+)\s+"
        r"(?P<teile>[\d.,]+)\s+"
        r"(?P<vol_ausgang>[\d.,]+)"
        r"(?P<extra>(?:\s+.*)?)$"
    )
    sub_row_pattern = re.compile(
        r"^(?P<unterauftrags_muster>\d+x\d+)\s+"
        r"(?P<teile>[\d.,]+)\s+"
        r"(?P<vol_ausgang>[\d.,]+)$"
    )
    
    merged_lines = []
    buffer = ""
    for line in lines:
        if sub_row_pattern.match(line):
            if buffer:
                if "Auftrag" in buffer:
                    m = re.search(r'\d{5}\s*-\s*', buffer)
                    if m:
                        buffer = buffer[m.start():]
                merged_lines.append(buffer)
                buffer = ""
            merged_lines.append(line)
            continue

        if re.match(r'^\d{5}\s*-\s*', line):
            if buffer:
                if "Auftrag" in buffer:
                    m = re.search(r'\d{5}\s*-\s*', buffer)
                    if m:
                        buffer = buffer[m.start():]
                merged_lines.append(buffer)
            buffer = line
        else:
            if buffer:
                buffer += " " + line
            else:
                buffer = line
    if buffer:
        if "Auftrag" in buffer:
            m = re.search(r'\d{5}\s*-\s*', buffer)
            if m:
                buffer = buffer[m.start():]
        merged_lines.append(buffer)
    
    result_rows = []
    current_main_order = None
    for line in merged_lines:
        line = line.strip()
        if not line:
            continue
        main_match = main_row_pattern.match(line)
        if main_match:
            main_dict = main_match.groupdict()
            auftrag = main_dict['auftrag'].strip()
            extra = main_dict.get('extra', '').strip()
            if extra:
                tokens = extra.split()
                if tokens and not re.fullmatch(r'[\d.,]+', tokens[-1]):
                    auftrag += " " + tokens[-1]
            current_main_order = main_dict
            current_main_order['auftrag'] = auftrag
            row = {
                'auftrag': current_main_order['auftrag'],
                'unterkategorie': "",
                'stämme': main_dict['stämme'],
                'vol_eingang': main_dict['vol_eingang'],
                'durchschn_stammlänge': main_dict['durchschn_stammlänge'],
                'teile': main_dict['teile'],
                'vol_ausgang': main_dict['vol_ausgang'],
            }
            result_rows.append(row)
        else:
            if current_main_order:
                sub_match = sub_row_pattern.match(line)
                if sub_match:
                    sub_row = sub_match.groupdict()
                    full_row = {
                        'auftrag': current_main_order['auftrag'],
                        'unterkategorie': sub_row['unterauftrags_muster'],
                        'stämme': "",
                        'vol_eingang': "",
                        'durchschn_stammlänge': "",
                        'teile': sub_row['teile'],
                        'vol_ausgang': sub_row['vol_ausgang'],
                    }
                    result_rows.append(full_row)
    return pd.DataFrame(result_rows)

################################################################################
# 3) Haupt-App: Finales Ergebnis (Direkt aggregiert, ohne separaten Zwischenschritt)
################################################################################
def main_app():
    st.title("Gelo Ausbeuteanalyse")

    # PDF-Upload
    st.markdown("### 1) PDF hochladen")
    pdf_file = st.file_uploader("PDF des Produktivitätsberichts", type=["pdf"])
    df_prod = None
    orders_from_pdf = {}
    auftrag_infos = {}
    if pdf_file:
        with st.spinner("Parsing PDF..."):
            df_prod = extract_table_with_suborders_clean(pdf_file)
        if df_prod is not None:
            st.subheader("Geparstes PDF")
            st.dataframe(df_prod)
            numeric_cols = ["stämme", "vol_eingang", "durchschn_stammlänge", "teile", "vol_ausgang"]
            for c in numeric_cols:
                df_prod[c] = pd.to_numeric(
                    df_prod[c].astype(str).str.replace(",", ".").str.strip(),
                    errors="coerce"
                )
            df_prod["unique_key"] = None
            df_main = df_prod[df_prod["unterkategorie"] == ""]
            for i, row in df_main.iterrows():
                order_text = row["auftrag"]
                ukey = f"{order_text}_{i}"
                df_prod.at[i, "unique_key"] = ukey
            df_prod["unique_key"] = df_prod["unique_key"].ffill()
            for i, row_ in df_main.iterrows():
                ukey = df_prod.at[i, "unique_key"]
                group = df_prod[df_prod["unique_key"] == ukey]
                dims = group["unterkategorie"].unique().tolist()
                dims = [d for d in dims if d]
                dims = [unify_dimension(normalize_dimension(d)) for d in dims]
                vol_in = 0.0
                main_line = group[group["unterkategorie"] == ""]
                if not main_line.empty:
                    vol_in_val = main_line.iloc[0]["vol_eingang"]
                    vol_in = parse_vol_eingang(vol_in_val)
                orders_from_pdf[ukey] = {"dimensions": dims, "auftrag": row_["auftrag"]}
                auftrag_infos[ukey] = {"vol_eingang": vol_in}

    # CSV-Upload
    st.markdown("### 2) MicroTec CSV hochladen")
    csv_file = st.file_uploader("CSV MicroTec", type=["csv"])
    df_microtec = None
    if csv_file:
        with st.spinner("Lade CSV..."):
            df_microtec = load_and_prepare_data(csv_file)
        if df_microtec is not None:
            st.subheader("MicroTec CSV Daten")
            st.dataframe(df_microtec.head())
    if not orders_from_pdf or df_microtec is None:
        st.stop()

    # Zeitfenster definieren
    default_date = df_microtec["Datetime"].iloc[0].date()
    st.markdown("### 3) Zeitfenster definieren")
    orders_final = {}
    order_instances = {}
    for ukey, info in orders_from_pdf.items():
        order_number = info["auftrag"].split()[0]
        order_instances.setdefault(order_number, []).append(ukey)
    for order_number, ukey_list in order_instances.items():
        for idx, ukey in enumerate(ukey_list):
            st.markdown(f"#### Auftrag {order_number} – Instanz {idx+1} von {len(ukey_list)}")
            c1, c2 = st.columns(2)
            sh = c1.number_input(f"Start-Stunde {ukey}", 0, 23, 6, 1)
            sm = c1.number_input(f"Start-Minute {ukey}", 0, 59, 0, 1)
            eh = c2.number_input(f"End-Stunde {ukey}", 0, 23, 12, 1)
            em = c2.number_input(f"End-Minute {ukey}", 0, 59, 0, 1)
            start_dt = datetime.combine(default_date, time(sh, sm))
            end_dt = datetime.combine(default_date, time(eh, em))
            orders_final[ukey] = {"time_window": (start_dt, end_dt), "dimensions": orders_from_pdf[ukey]["dimensions"]}

    def compute_yield(volume, vol_in_liters):
        if vol_in_liters == 0:
            return 0
        return (volume / (vol_in_liters / 1000)) * 100

    # Finaler Aggregationsschritt (ohne separaten Zwischenschritt)
    if st.button("Auswerten & Aggregieren"):
        all_rows = []
        for ukey, params in orders_final.items():
            (start_dt, end_dt) = params["time_window"]
            dims = params["dimensions"]
            df_filtered = filter_data_for_order(df_microtec, start_dt, end_dt, dims)
            result = summarize_cbm_by_classifications(df_filtered)
            vol_in = auftrag_infos[ukey]["vol_eingang"]
            for _, row_ in result.iterrows():
                dim = row_["Dimension"]
                brutto_vol = row_["total_cbm"]
                waste_vol = row_["waste_cbm"]
                ce = row_["ce_cbm"]
                kh_i_iii = row_["kh_i_iii_cbm"]
                sf_i_iii = row_["sf_i_iii_cbm"]
                sf_i_iiii = row_["sf_i_iiii_cbm"]
                si_0_iv = row_["si_0_iv_cbm"]
                si_i_ii = row_["si_i_ii_cbm"]
                ind_ii_iii = row_["ind_ii_iii_cbm"]
                nsi_i_iii = row_["nsi_i_iii_cbm"]
                ass_iv = row_["ass_iv_cbm"]
                waste_pct = row_["waste_percent"]
                netto_vol = brutto_vol - waste_vol
                brutto_ausb = compute_yield(brutto_vol, vol_in)
                netto_ausb = compute_yield(netto_vol, vol_in)
                all_rows.append({
                    "unique_key": ukey,
                    "unterkategorie": dim,
                    "Brutto_Volumen": brutto_vol,
                    "waste_cbm": waste_vol,
                    "Netto_Volumen": netto_vol,
                    "Brutto_Ausbeute": brutto_ausb,
                    "Netto_Ausbeute": netto_ausb,
                    "Vol_Eingang_m3": vol_in / 1000,
                    "Brutto_Ausschuss": waste_pct,
                    "ce_cbm": ce,
                    "kh_i_iii_cbm": kh_i_iii,
                    "sf_i_iii_cbm": sf_i_iii,
                    "sf_i_iiii_cbm": sf_i_iiii,
                    "si_0_iv_cbm": si_0_iv,
                    "si_i_ii_cbm": si_i_ii,
                    "ind_ii_iii_cbm": ind_ii_iii,
                    "nsi_i_iii_cbm": nsi_i_iii,
                    "ass_iv_cbm": ass_iv,
                })
        microtec_df = pd.DataFrame(all_rows)

        # Merge mit PDF-Daten
        df_prod["unterkategorie"] = df_prod["unterkategorie"].apply(normalize_dimension).apply(unify_dimension)
        merged_rows = []
        for _, pdfrow in df_prod.iterrows():
            ukey = pdfrow["unique_key"]
            ukat = pdfrow["unterkategorie"]
            row_dict = pdfrow.to_dict()
            if ukey is not None:
                match = microtec_df.loc[
                    (microtec_df["unique_key"] == ukey) & 
                    (microtec_df["unterkategorie"] == ukat)
                ]
            else:
                match = pd.DataFrame()
            if not match.empty:
                rowM = match.iloc[0]
                row_dict["Brutto_Volumen"] = rowM["Brutto_Volumen"]
                row_dict["waste_cbm"] = rowM["waste_cbm"]
                row_dict["Netto_Volumen"] = rowM["Netto_Volumen"]
                row_dict["Brutto_Ausbeute"] = rowM["Brutto_Ausbeute"]
                row_dict["Netto_Ausbeute"] = rowM["Netto_Ausbeute"]
                row_dict["Vol_Eingang_m3"] = rowM["Vol_Eingang_m3"]
                row_dict["Brutto_Ausschuss"] = rowM["Brutto_Ausschuss"]
                row_dict["ce_cbm"] = rowM["ce_cbm"]
                row_dict["kh_i_iii_cbm"] = rowM["kh_i_iii_cbm"]
                row_dict["sf_i_iii_cbm"] = rowM["sf_i_iii_cbm"]
                row_dict["sf_i_iiii_cbm"] = rowM["sf_i_iiii_cbm"]
                row_dict["si_0_iv_cbm"] = rowM["si_0_iv_cbm"]
                row_dict["si_i_ii_cbm"] = rowM["si_i_ii_cbm"]
                row_dict["ind_ii_iii_cbm"] = rowM["ind_ii_iii_cbm"]
                row_dict["nsi_i_iii_cbm"] = rowM["nsi_i_iii_cbm"]
                row_dict["ass_iv_cbm"] = rowM["ass_iv_cbm"]
            else:
                row_dict["Brutto_Volumen"] = 0
                row_dict["waste_cbm"] = 0
                row_dict["Netto_Volumen"] = 0
                row_dict["Brutto_Ausbeute"] = 0
                row_dict["Netto_Ausbeute"] = 0
                row_dict["Vol_Eingang_m3"] = 0
                row_dict["Brutto_Ausschuss"] = 0
                row_dict["ce_cbm"] = 0
                row_dict["kh_i_iii_cbm"] = 0
                row_dict["sf_i_iii_cbm"] = 0
                row_dict["sf_i_iiii_cbm"] = 0
                row_dict["si_0_iv_cbm"] = 0
                row_dict["si_i_ii_cbm"] = 0
                row_dict["ind_ii_iii_cbm"] = 0
                row_dict["nsi_i_iii_cbm"] = 0
                row_dict["ass_iv_cbm"] = 0
            merged_rows.append(row_dict)
        merged_df = pd.DataFrame(merged_rows)
        st.session_state["merged_df"] = merged_df

        # Aggregationsschritt
        df_agg = st.session_state["merged_df"].copy()
        numeric_cols = [
            "stämme", "vol_eingang", "durchschn_stammlänge", "teile", "vol_ausgang",
            "Brutto_Volumen", "Brutto_Ausschuss", "Netto_Volumen", "Brutto_Ausbeute",
            "Netto_Ausbeute", "Vol_Eingang_m3",
            "ce_cbm", "kh_i_iii_cbm", "sf_i_iii_cbm", "sf_i_iiii_cbm",
            "si_0_iv_cbm", "si_i_ii_cbm", "ind_ii_iii_cbm", "nsi_i_iii_cbm", "ass_iv_cbm", "waste_cbm"
        ]
        for c in numeric_cols:
            if c not in df_agg.columns:
                df_agg[c] = 0
            df_agg[c] = pd.to_numeric(df_agg[c], errors="coerce")
        agg_dict = {
            "stämme": "sum",
            "vol_eingang": "sum",
            "durchschn_stammlänge": "mean",
            "teile": "sum",
            "vol_ausgang": "sum",
            "Brutto_Volumen": "sum",
            "Brutto_Ausschuss": "mean",
            "Netto_Volumen": "sum",
            "Vol_Eingang_m3": "sum",
            "Brutto_Ausbeute": "mean",
            "Netto_Ausbeute": "mean",
            "ce_cbm": "sum",
            "kh_i_iii_cbm": "sum",
            "sf_i_iii_cbm": "sum",
            "sf_i_iiii_cbm": "sum",
            "si_0_iv_cbm": "sum",
            "si_i_ii_cbm": "sum",
            "ind_ii_iii_cbm": "sum",
            "nsi_i_iii_cbm": "sum",
            "ass_iv_cbm": "sum",
            "waste_cbm": "sum"
        }
        grouped = df_agg.groupby(["auftrag", "unterkategorie"], as_index=False).agg(agg_dict)
        grouped["Brutto_Ausschuss"] = grouped.apply(
            lambda r: round(100 * (r["waste_cbm"] / r["Brutto_Volumen"]), 3) if r["Brutto_Volumen"] > 0 else 0,
            axis=1
        )
        grouped["Netto_Volumen"] = grouped["Brutto_Volumen"] - grouped["waste_cbm"]
        def compute_yield(row, colname):
            vol_in_liters = row["Vol_Eingang_m3"]
            if vol_in_liters == 0:
                return 0
            return (row[colname] / vol_in_liters) * 100
        grouped["Brutto_Ausbeute"] = grouped.apply(lambda r: round(compute_yield(r, "Brutto_Volumen"), 3), axis=1)
        grouped["Netto_Ausbeute"]  = grouped.apply(lambda r: round(compute_yield(r, "Netto_Volumen"), 3), axis=1)
        # ---------------------------------------------------------------------
        # Zusammenfassen der SF-Spalten: KH I-III + SF I-III + SF I-IV -> sf_cbm
        grouped["sf_cbm"] = grouped["kh_i_iii_cbm"] + grouped["sf_i_iii_cbm"] + grouped["sf_i_iiii_cbm"]
        grouped.drop(["kh_i_iii_cbm", "sf_i_iii_cbm", "sf_i_iiii_cbm"], axis=1, inplace=True)
        # ---------------------------------------------------------------------
        # Zusammenfassen der SI-Spalten: SI 0-IV + SI I-II -> si_cbm
        grouped["si_cbm"] = grouped["si_0_iv_cbm"] + grouped["si_i_ii_cbm"]
        grouped.drop(["si_0_iv_cbm", "si_i_ii_cbm"], axis=1, inplace=True)
        final_cols = [
            "auftrag", "unterkategorie", "stämme", "vol_eingang", "durchschn_stammlänge",
            "teile", "vol_ausgang", "Brutto_Volumen", "Brutto_Ausschuss",
            "Netto_Volumen", "Brutto_Ausbeute", "Netto_Ausbeute",
            "ce_cbm", "sf_cbm", "si_cbm", "ind_ii_iii_cbm", "nsi_i_iii_cbm", "ass_iv_cbm", "waste_cbm"
        ]
        for col in final_cols:
            if col not in grouped.columns:
                grouped[col] = 0
        grouped = grouped[final_cols]
        # Spalten-Umbennung im finalen Output
        rename_map = {
            "auftrag": "Auftrag",
            "unterkategorie": "Dimension",
            "stämme": "Stämme",
            "vol_eingang": "Volumen_Eingang",
            "durchschn_stammlänge": "Durchschn_Stämme",
            "teile": "Teile",
            "vol_ausgang": "Volumen_Ausgang",
            "Brutto_Volumen": "Brutto_Volumen",
            "Brutto_Ausschuss": "Brutto_Ausschuss",
            "Netto_Volumen": "Netto_Volumen",
            "Brutto_Ausbeute": "Brutto_Ausbeute",
            "Netto_Ausbeute": "Netto_Ausbeute",
            "ce_cbm": "CE",
            "sf_cbm": "SF",
            "si_cbm": "SI",
            "ind_ii_iii_cbm": "IND",
            "nsi_i_iii_cbm": "NSI",
            "ass_iv_cbm": "Q_V",
            "waste_cbm": "Ausschuss"
        }
        grouped.rename(columns=rename_map, inplace=True)

        # Final: Transformation der Dimensionen
        final_dimension_map = {
            "17x100": "17x98",
            "23x103": "23x100",
            "47x221": "47x220",
            "47x223": "47x220"
        }
        grouped["Dimension"] = grouped["Dimension"].replace(final_dimension_map)

        st.subheader("Aggregiertes Ergebnis")
        st.dataframe(grouped)
        xlsx_data = to_excel_bytes(grouped)
        st.download_button(
            label="Download Aggregiertes Ergebnis",
            data=xlsx_data,
            file_name=f"Ausbeuteanalyse_{default_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main_app()
