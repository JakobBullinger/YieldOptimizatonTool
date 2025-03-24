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
# 2) Deine bisherigen Hilfsfunktionen: Synonym, CSV-Funktionen, PDF-Parsing
################################################################################

synonyms = {
    "38x80": "80x80",
    "17x75": "17x78",
    "17x98": "17x100",
    "17x95": "17x100",
    "23x100": "23x103",
    "47x220": "47x221"
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

def summarize_cbm_and_waste(df):
    if df.empty:
        return pd.DataFrame(columns=['Dimension', 'total_cbm', 'waste_cbm', 'waste_percent'])
    grouped = df.groupby('Dimension').agg(
        total_cbm=('CBM', 'sum'),
        waste_cbm=('CBM', lambda x: x[df.loc[x.index, 'Classification'].str.lower() == 'waste'].sum())
    ).reset_index()
    grouped['waste_percent'] = (grouped['waste_cbm'] / grouped['total_cbm'] * 100).round(2)
    grouped['total_cbm'] = grouped['total_cbm'].round(3)
    grouped['waste_cbm'] = grouped['waste_cbm'].round(3)
    return grouped

# Klassifizierungs-Listen (HW/SW/KH) + classify_dimension
HW_DIMENSIONS = {
    "47x135","47x198","47x220","47x221",
    "43x95","43x138","43x125","43x141","47x92","47x113",
    "47x153","47x154","47x175","47x198","51x104","36x134",
    "48x117","48x128","48x144","48x164","69x122","75x148","75x152",
    "46x88"
}
SW_DIMENSIONS = {
    "17x75","17x78","17x98","17x100","23x103","26x146","28x131","28x149","35x155"
}
KH_DIMENSIONS = {
    "56x76","58x78","60x80","60x90","63x80","65x80","68x98","70x75","75x90","75x95",
    "75x98","76x76","76x96","78x78","78x90","78x100","79x100","93x93","96x116","98x98","33x90"
}

def classify_dimension(dim):
    if not isinstance(dim, str):
        return "Unknown"
    dim_lower = dim.lower().strip()
    if dim_lower in (d.lower() for d in HW_DIMENSIONS):
        return "HW"
    elif dim_lower in (d.lower() for d in SW_DIMENSIONS):
        return "SW"
    elif dim_lower in (d.lower() for d in KH_DIMENSIONS):
        return "KH"
    else:
        return "Unknown"

# PDF-Parsing
import pdfplumber

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

def make_invisible_key(order_name, idx):
    return order_name + ("\u200B" * idx)


################################################################################
# 3) Haupt-App: Zwei Buttons (Schritt 5 & Schritt 6)
################################################################################
def main_app():
    st.title("PDF + MicroTec Auswertung: Doppelte Einträge, neu berechnete Kennzahlen")

    # -------------------------------------------------------------------------
    # PDF-Upload
    # -------------------------------------------------------------------------
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

            # Numeric cast
            numeric_cols = ["stämme","vol_eingang","durchschn_stammlänge","teile","vol_ausgang"]
            for c in numeric_cols:
                df_prod[c] = pd.to_numeric(
                    df_prod[c].astype(str).str.replace(",",".").str.strip(),
                    errors="coerce"
                )

            # "Hauptzeilen" => unique_keys
            df_main = df_prod[df_prod["unterkategorie"]==""].reset_index(drop=True)
            for i,row_ in df_main.iterrows():
                order = row_["auftrag"]
                group = df_prod[df_prod["auftrag"]==order]
                dims  = group["unterkategorie"].unique().tolist()
                dims  = [d for d in dims if d]
                dims  = [unify_dimension(normalize_dimension(d)) for d in dims]

                # vol_in
                main_line = group[group["unterkategorie"]==""]
                vol_in = 0.0
                if not main_line.empty:
                    vol_in_val = main_line.iloc[0]["vol_eingang"]
                    vol_in = parse_vol_eingang(vol_in_val)

                # zero-width key
                ukey = make_invisible_key(order, i)
                orders_from_pdf[ukey] = dims
                auftrag_infos[ukey]   = {"vol_eingang": vol_in}

    # -------------------------------------------------------------------------
    # CSV-Upload
    # -------------------------------------------------------------------------
    st.markdown("### 2) MicroTec CSV hochladen")
    csv_file = st.file_uploader("CSV MicroTec", type=["csv"])
    df_microtec = None

    if csv_file:
        with st.spinner("Lade CSV..."):
            df_microtec = load_and_prepare_data(csv_file)
        if df_microtec is not None:
            st.subheader("MicroTec CSV Daten")
            st.dataframe(df_microtec.head())

    if not orders_from_pdf:
        st.stop()
    if df_microtec is None:
        st.stop()

    # -------------------------------------------------------------------------
    # Zeitfenster
    # -------------------------------------------------------------------------
    default_date = df_microtec["Datetime"].iloc[0].date()
    st.markdown("### 3) Zeitfenster definieren")
    orders_final={}
    for ukey in orders_from_pdf.keys():
        st.markdown(f"#### Auftrag: {ukey}")
        c1,c2= st.columns(2)
        sh = c1.number_input(f"Start-Stunde {ukey}", 0,23,6,1)
        sm = c1.number_input(f"Start-Minute {ukey}", 0,59,0,1)
        eh = c2.number_input(f"End-Stunde {ukey}",   0,23,12,1)
        em = c2.number_input(f"End-Minute {ukey}",   0,59,0,1)
        start_dt = datetime.combine(default_date, time(sh,sm))
        end_dt   = datetime.combine(default_date, time(eh,em))

        orders_final[ukey] = {
            "time_window": (start_dt,end_dt),
            "dimensions": orders_from_pdf[ukey]
        }

    # Ausbeuten-Funktion
    def compute_yield(volume, vol_in_liters):
        if vol_in_liters==0:
            return 0
        return (volume/(vol_in_liters/1000))*100

    # -------------------------------------------------------------------------
    # BUTTON (A): Auswerten & Zusammenführen (nicht aggregiert)
    # -------------------------------------------------------------------------
    if st.button("Auswerten & Zusammenführen (ohne Aggregation)"):
        final_rows=[]
        for ukey, params in orders_final.items():
            (start_dt, end_dt) = params["time_window"]
            dims = params["dimensions"]
            df_filtered = filter_data_for_order(df_microtec, start_dt, end_dt, dims)
            result = summarize_cbm_and_waste(df_filtered)
            vol_in = auftrag_infos[ukey]["vol_eingang"]

            for _, row_ in result.iterrows():
                dim        = row_["Dimension"]
                brutto_vol = row_["total_cbm"]  # Roh: total_cbm
                waste_vol  = row_["waste_cbm"]
                waste_pct  = row_["waste_percent"]

                netto_vol  = brutto_vol - waste_vol
                brutto_ausb = compute_yield(brutto_vol, vol_in)
                netto_ausb  = compute_yield(netto_vol,  vol_in)
                wtyp        = classify_dimension(dim)

                # NEU: Wir speichern hier EXPLIZIT "waste_cbm", 
                # damit wir in Schritt 6 den summierten Wert haben.
                final_rows.append({
                    "unique_key": ukey,
                    "unterkategorie": dim,
                    "Brutto_Volumen": brutto_vol,
                    "waste_cbm": waste_vol,  # <-- NEU
                    "Netto_Volumen": netto_vol,
                    "Brutto_Ausbeute": brutto_ausb,
                    "Netto_Ausbeute": netto_ausb,
                    "Vol_Eingang_m3": vol_in/1000,
                    "Brutto_Ausschuss": waste_pct,  # momentan nur Einzellauf
                    "Warentyp": wtyp
                })

        microtec_df = pd.DataFrame(final_rows)
        st.subheader("MicroTec-Ergebnis (nicht aggregiert)")
        st.dataframe(microtec_df)

        # Merge mit df_prod
        df_prod["unterkategorie"] = df_prod["unterkategorie"].apply(normalize_dimension).apply(unify_dimension)
        df_prod["auftrag_num"]    = df_prod["auftrag"].str.extract(r'^(\d{5})', expand=False)

        microtec_df["auftrag_num"] = microtec_df["unique_key"].apply(lambda x: x[:5])

        merged_rows=[]
        for _, pdfrow in df_prod.iterrows():
            a_num = pdfrow["auftrag_num"]
            ukat  = pdfrow["unterkategorie"]
            match = microtec_df.loc[
                (microtec_df["auftrag_num"]==a_num) & 
                (microtec_df["unterkategorie"]==ukat)
            ]
            row_dict = pdfrow.to_dict()

            if not match.empty:
                # Nimm die erste Zeile
                rowM = match.iloc[0]
                row_dict["Brutto_Volumen"]   = rowM["Brutto_Volumen"]
                row_dict["waste_cbm"]        = rowM["waste_cbm"]
                row_dict["Netto_Volumen"]    = rowM["Netto_Volumen"]
                row_dict["Brutto_Ausbeute"]  = rowM["Brutto_Ausbeute"]
                row_dict["Netto_Ausbeute"]   = rowM["Netto_Ausbeute"]
                row_dict["Vol_Eingang_m3"]   = rowM["Vol_Eingang_m3"]
                row_dict["Brutto_Ausschuss"] = rowM["Brutto_Ausschuss"]
                row_dict["Warentyp"]         = rowM["Warentyp"]
            else:
                # leere Felder
                row_dict["Brutto_Volumen"]   = 0
                row_dict["waste_cbm"]        = 0
                row_dict["Netto_Volumen"]    = 0
                row_dict["Brutto_Ausbeute"]  = 0
                row_dict["Netto_Ausbeute"]   = 0
                row_dict["Vol_Eingang_m3"]   = 0
                row_dict["Brutto_Ausschuss"] = 0
                row_dict["Warentyp"]         = ""

            merged_rows.append(row_dict)

        merged_df = pd.DataFrame(merged_rows)

        st.session_state["merged_df"] = merged_df
        st.subheader("Nicht aggregiertes Endergebnis (PDF + MicroTec)")
        st.dataframe(merged_df)

    # -------------------------------------------------------------------------
    # BUTTON (B): Aggregieren => Summen bilden & Kennzahlen NEU berechnen
    # -------------------------------------------------------------------------
    if st.button("Aggregieren & Kennzahlen neu"):
        if st.session_state["merged_df"] is None:
            st.error("Bitte erst 'Auswerten & Zusammenführen' klicken!")
            st.stop()

        df_agg = st.session_state["merged_df"].copy()

        # 1) numeric cast
        numeric_cols = [
            "stämme","vol_eingang","durchschn_stammlänge","teile","vol_ausgang",
            "Brutto_Volumen","waste_cbm","Netto_Volumen","Brutto_Ausbeute",
            "Netto_Ausbeute","Vol_Eingang_m3","Brutto_Ausschuss"
        ]
        for c in numeric_cols:
            if c not in df_agg.columns:
                df_agg[c] = 0
            df_agg[c] = pd.to_numeric(df_agg[c], errors="coerce")

            # #2) Debug-Ausgabe je Gruppe

            # st.write("## Debug-Ausgabe vor groupby")
            # group_cols = ["auftrag", "unterkategorie"]  # oder was du verwendest
            # for (auftrag_val, unterkat_val), subdf in df_agg.groupby(group_cols):
            #     st.write(f"### Gruppe: {auftrag_val}, {unterkat_val}")
            #     st.write(subdf[
            #         ["Brutto_Volumen", "waste_cbm", "Netto_Volumen", 
            #         "vol_eingang", "Vol_Eingang_m3", "Brutto_Ausbeute", "Netto_Ausbeute"]
            #     ])
            #     st.write("Summen: ")
            #     st.write({
            #         "Sum_Brutto": subdf["Brutto_Volumen"].sum(),
            #         "Sum_waste":  subdf["waste_cbm"].sum(),
            #         "Sum_Netto":  subdf["Netto_Volumen"].sum(),
            #         "Sum_vol_in": subdf["vol_eingang"].sum(),
            #         # usw.
            #     })
            #     st.write("--------")
        agg_dict = {
            "stämme": "sum",
            "vol_eingang": "sum",  # in L
            "durchschn_stammlänge": "mean",
            "teile": "sum",
            "vol_ausgang": "sum",
            "Brutto_Volumen": "sum",
            "waste_cbm": "sum",  # neu
            "Netto_Volumen": "sum",  # wir überschreiben es gleich
            "Vol_Eingang_m3": "sum", # wir überschreiben es gleich
            "Brutto_Ausbeute": "mean",  # wir überschreiben es
            "Netto_Ausbeute": "mean",   # wir überschreiben es
            "Brutto_Ausschuss": "mean", # wir überschreiben es
            "Warentyp": "first"         # je nach Bedarf
        }

        grouped = (
            df_agg
            .groupby(["auftrag","unterkategorie"], as_index=False)
            .agg(agg_dict)
        )

        # 3) NEU berechnen:
        # => brutto_ausschuss = ( sum(waste_cbm) / sum(Brutto_Volumen) )*100
        # => Netto_Volumen = sum(Brutto_Volumen) - sum(waste_cbm)
        # => sum(vol_in_liters) = "vol_eingang" (in L)
        # => Brutto_Ausbeute = (Brutto_Volumen / (vol_eingang/1000))*100
        # => Netto_Ausbeute  = (Netto_Volumen  / (vol_eingang/1000))*100

        grouped["Brutto_Ausschuss"] = grouped.apply(
            lambda r: 100*(r["waste_cbm"]/r["Brutto_Volumen"]) if r["Brutto_Volumen"]>0 else 0,
            axis=1
        )
        grouped["Netto_Volumen"] = grouped["Brutto_Volumen"] - grouped["waste_cbm"]

        def compute_ausbeute(row, colname):
            vol_in_liters = row["Vol_Eingang_m3"]
            if vol_in_liters==0:
                return 0
            return (row[colname]/(vol_in_liters))*100

        grouped["Brutto_Ausbeute"] = grouped.apply(lambda r: compute_ausbeute(r,"Brutto_Volumen"), axis=1)
        grouped["Netto_Ausbeute"]  = grouped.apply(lambda r: compute_ausbeute(r,"Netto_Volumen"),  axis=1)

        # # Vol_Eingang_m3 = sum(vol_eingang)/1000
        # grouped["Vol_Eingang_m3"] = grouped["vol_eingang"]/1000.0

        # 4) Spalten in gewünschter Reihenfolge
        final_cols = [
            "auftrag","unterkategorie","stämme","vol_eingang","durchschn_stammlänge",
            "teile","vol_ausgang","Warentyp","Brutto_Volumen","Brutto_Ausschuss",
            "Netto_Volumen","Brutto_Ausbeute","Netto_Ausbeute","Vol_Eingang_m3"
        ]
        for col in final_cols:
            if col not in grouped.columns:
                grouped[col] = 0
        grouped = grouped[final_cols]

        st.subheader("Aggregiertes Ergebnis (Brutto_Ausschuss, Ausbeute etc. NEU)")
        st.dataframe(grouped)

        xlsx_data = to_excel_bytes(grouped)
        st.download_button(
            label="Download Aggregiertes Ergebnis",
            data=xlsx_data,
            file_name="Aggregiertes_Ergebnis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__=="__main__":
    main_app()
