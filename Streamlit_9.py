###############################################################################
# Streamlit-App: Gelo Ausbeuteanalyse
# (einzige Änderungen: korrekte Volumen-Einlesung + Ausbeute-Berechnung)
###############################################################################
import streamlit as st
import pandas as pd
import pdfplumber
import re
from datetime import datetime, time
import io
import math

################################################################################
# 1) Session-State
################################################################################
if "merged_df" not in st.session_state:
    st.session_state["merged_df"] = None

################################################################################
# 2) Hilfsfunktionen
################################################################################
# ---------- Dimensions-Synonyme ----------
synonyms = {
    "38x80": "80x80",
    "17x75": "17x78",
    "17x98": "17x100",
    "17x95": "17x100",
    "23x100": "23x103",
    "47x220": "47x223",
    "47x221": "47x222",
}

def unify_dimension(dim_str):
    return synonyms.get(dim_str, dim_str)

def normalize_dimension(dim_str):
    if pd.isna(dim_str):
        return "Unknown"
    dim_str = (
        str(dim_str)
        .lower()
        .replace("mm", "")
        .replace(",", ".")
        .replace(" ", "")
        .replace(".0", "")
    )
    return dim_str

# ---------- KORRIGIERT: Volumen-Parser (liefert m³) ----------
def parse_vol_eingang(vol_str):
    """
    Konvertiert eine Volumenzahl (de/​en Schreibweisen) in **m³**.
    Beispiele:
        "237.682"  -> 237.682
        "237,682"  -> 237.682
        "1.234,56" -> 1234.56
    """
    if pd.isna(vol_str) or str(vol_str).strip() == "":
        return 0.0

    s = str(vol_str).strip()

    # Tausenderpunkt + Dezimalkomma
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    # Nur Komma als Dezimaltrennzeichen
    elif "," in s:
        s = s.replace(",", ".")
    # Nur Punkt => already fine

    try:
        return float(s)             # ***m³***
    except ValueError:
        return 0.0

def to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Auswertung")
    return output.getvalue()

# ---------- CSV-Einlesen ----------
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, header=None, sep=";")
    df.rename(
        columns={
            1: "Year",
            2: "Month",
            3: "Day",
            4: "Hour",
            5: "Minute",
            15: "Dimension",
            21: "Classification",
            26: "CBM",
        },
        inplace=True,
    )
    df["Datetime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df["Dimension"] = (
        df["Dimension"].apply(normalize_dimension).apply(unify_dimension)
    )
    df["CBM"] = pd.to_numeric(df["CBM"], errors="coerce")
    return df

def filter_data_for_order(df, start_dt, end_dt, dimensions):
    mask = (df["Datetime"] >= start_dt) & (df["Datetime"] < end_dt) & (
        df["Dimension"].isin(dimensions)
    )
    return df.loc[mask].copy()

# ---------- Summieren der CBM ----------
def summarize_cbm_by_classifications(df):
    CLASSIFICATION_MAP = {
        "Waste": "waste_cbm",
        "CE": "ce_cbm",
        "KH I-III": "kh_i_iii_cbm",
        "SF I-III": "sf_i_iii_cbm",
        "SF I-IV": "sf_i_iiii_cbm",
        "SI 0-IV": "si_0_iv_cbm",
        "SI I-II": "si_i_ii_cbm",
        "IND II-III": "ind_ii_iii_cbm",
        " NSI I-III": "nsi_i_iii_cbm",
        "ASS IV": "ass_iv_cbm",
    }
    grouped = df.groupby("Dimension", dropna=False).agg(total_cbm=("CBM", "sum")).reset_index()

    for class_label, col_name in CLASSIFICATION_MAP.items():
        grouped[col_name] = grouped["Dimension"].apply(
            lambda dim: df.loc[
                (df["Dimension"] == dim)
                & (df["Classification"].str.lower() == class_label.lower()),
                "CBM",
            ].sum()
        )

    grouped["waste_percent"] = grouped.apply(
        lambda row: 100 * row["waste_cbm"] / row["total_cbm"] if row["total_cbm"] else 0,
        axis=1,
    )

    grouped["total_cbm"] = grouped["total_cbm"].round(3)
    for col_name in CLASSIFICATION_MAP.values():
        grouped[col_name] = grouped[col_name].round(3)
    grouped["waste_percent"] = grouped["waste_percent"].round(2)
    return grouped

# ---------- PDF-Parsing (unverändert) ----------
def extract_table_with_suborders_clean(file_input, start_keyword="Auftrag"):
    # ... (Code wie gehabt, unverändert) ...
    # Der komplette Funktionskörper bleibt unverändert
    # -------------------------------------------------------------------------
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
                    m = re.search(r"\d{5}\s*-\s*", buffer)
                    if m:
                        buffer = buffer[m.start():]
                merged_lines.append(buffer)
                buffer = ""
            merged_lines.append(line)
            continue

        if re.match(r"^\d{5}\s*-\s*", line):
            if buffer:
                if "Auftrag" in buffer:
                    m = re.search(r"\d{5}\s*-\s*", buffer)
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
            m = re.search(r"\d{5}\s*-\s*", buffer)
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
            auftrag = main_dict["auftrag"].strip()
            extra = main_dict.get("extra", "").strip()
            if extra:
                tokens = extra.split()
                if tokens and not re.fullmatch(r"[\d.,]+", tokens[-1]):
                    auftrag += " " + tokens[-1]

            current_main_order = main_dict
            current_main_order["auftrag"] = auftrag
            row = {
                "auftrag": current_main_order["auftrag"],
                "unterkategorie": "",
                "stämme": main_dict["stämme"],
                "vol_eingang": main_dict["vol_eingang"],
                "durchschn_stammlänge": main_dict["durchschn_stammlänge"],
                "teile": main_dict["teile"],
                "vol_ausgang": main_dict["vol_ausgang"],
            }
            result_rows.append(row)
        else:
            if current_main_order:
                sub_match = sub_row_pattern.match(line)
                if sub_match:
                    sub_row = sub_match.groupdict()
                    full_row = {
                        "auftrag": current_main_order["auftrag"],
                        "unterkategorie": sub_row["unterauftrags_muster"],
                        "stämme": "",
                        "vol_eingang": "",
                        "durchschn_stammlänge": "",
                        "teile": sub_row["teile"],
                        "vol_ausgang": sub_row["vol_ausgang"],
                    }
                    result_rows.append(full_row)
    return pd.DataFrame(result_rows)

################################################################################
# 3) Haupt-App
################################################################################
def main_app():
    st.title("Gelo Ausbeuteanalyse")

    # ---------- PDF-Upload ----------
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

            numeric_cols = [
                "stämme",
                "vol_eingang",
                "durchschn_stammlänge",
                "teile",
                "vol_ausgang",
            ]
            for c in numeric_cols:
                df_prod[c] = pd.to_numeric(
                    df_prod[c].astype(str).str.replace(",", ".").str.strip(),
                    errors="coerce",
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

                # Dimensionen der Unterkategorien
                dims = group["unterkategorie"].unique().tolist()
                dims = [d for d in dims if d]
                dims = [unify_dimension(normalize_dimension(d)) for d in dims]

                # Volumen Eingang (m³)
                main_line = group[group["unterkategorie"] == ""]
                vol_in = 0.0
                if not main_line.empty:
                    vol_in_val = main_line.iloc[0]["vol_eingang"]
                    vol_in = parse_vol_eingang(vol_in_val)  # ***m³***

                orders_from_pdf[ukey] = {"dimensions": dims, "auftrag": row_["auftrag"]}
                auftrag_infos[ukey] = {"vol_eingang": vol_in}

    # ---------- CSV-Upload ----------
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

    # ---------- Zeitfenster je Auftrag ----------
    default_date = df_microtec["Datetime"].iloc[0].date()
    st.markdown("### 3) Zeitfenster definieren pro Auftrag")

    orders_time = {}
    order_instances = {}
    for ukey, info in orders_from_pdf.items():
        order_number = info["auftrag"].split()[0]
        order_instances.setdefault(order_number, []).append(ukey)

    for order_number, ukey_list in order_instances.items():
        for idx, ukey in enumerate(ukey_list):
            st.markdown(f"#### Auftrag {order_number} – Instanz {idx+1} von {len(ukey_list)}")
            st.markdown("##### Wirkliche Zeit (Laufzeit)")
            c1, c2 = st.columns(2)
            rt_sh = c1.number_input(f"Start-Stunde (Real) {ukey}", 0, 23, 6, 1)
            rt_sm = c1.number_input(f"Start-Minute (Real) {ukey}", 0, 59, 0, 1)
            rt_eh = c2.number_input(f"End-Stunde (Real) {ukey}", 0, 23, 12, 1)
            rt_em = c2.number_input(f"End-Minute (Real) {ukey}", 0, 59, 0, 1)
            runtime_start = datetime.combine(default_date, time(rt_sh, rt_sm))
            runtime_end = datetime.combine(default_date, time(rt_eh, rt_em))

            st.markdown("##### MicroTec Zeit")
            c3, c4 = st.columns(2)
            mt_sh = c3.number_input(f"Start-Stunde (MicroTec) {ukey}", 0, 23, 6, 1)
            mt_sm = c3.number_input(f"Start-Minute (MicroTec) {ukey}", 0, 59, 0, 1)
            mt_eh = c4.number_input(f"End-Stunde (MicroTec) {ukey}", 0, 23, 12, 1)
            mt_em = c4.number_input(f"End-Minute (MicroTec) {ukey}", 0, 59, 0, 1)
            microtec_start = datetime.combine(default_date, time(mt_sh, mt_sm))
            microtec_end = datetime.combine(default_date, time(mt_eh, mt_em))

            orders_time[ukey] = {
                "runtime": (runtime_start, runtime_end),
                "microtec": (microtec_start, microtec_end),
            }

    # ---------- KORRIGIERT: Ausbeute-Funktion (m³ -> %) ----------
    def compute_yield(volume_m3: float, vol_in_m3: float) -> float:
        if vol_in_m3 == 0:
            return 0.0
        return (volume_m3 / vol_in_m3) * 100

    # ---------- Aggregieren & Auswerten ----------
    if st.button("Auswerten & Aggregieren"):
        all_rows = []

        for ukey, times in orders_time.items():
            microtec_window = times["microtec"]
            (start_dt, end_dt) = microtec_window
            dims = orders_from_pdf[ukey]["dimensions"]

            df_filtered = filter_data_for_order(df_microtec, start_dt, end_dt, dims)
            result = summarize_cbm_by_classifications(df_filtered)

            vol_in = auftrag_infos[ukey]["vol_eingang"]  # ***m³***

            for _, row_ in result.iterrows():
                dim = row_["Dimension"]
                brutto_vol = row_["total_cbm"]
                waste_vol = row_["waste_cbm"]

                netto_vol = brutto_vol - waste_vol
                brutto_ausb = compute_yield(brutto_vol, vol_in)
                netto_ausb = compute_yield(netto_vol, vol_in)

                all_rows.append(
                    {
                        "unique_key": ukey,
                        "unterkategorie": dim,
                        "Brutto_Volumen": brutto_vol,
                        "waste_cbm": waste_vol,
                        "Netto_Volumen": netto_vol,
                        "Brutto_Ausbeute": brutto_ausb,
                        "Netto_Ausbeute": netto_ausb,
                        "Vol_Eingang_m3": vol_in,   # ***m³ ohne /1000***
                        "Brutto_Ausschuss": row_["waste_percent"],
                        "ce_cbm": row_["ce_cbm"],
                        "kh_i_iii_cbm": row_["kh_i_iii_cbm"],
                        "sf_i_iii_cbm": row_["sf_i_iii_cbm"],
                        "sf_i_iiii_cbm": row_["sf_i_iiii_cbm"],
                        "si_0_iv_cbm": row_["si_0_iv_cbm"],
                        "si_i_ii_cbm": row_["si_i_ii_cbm"],
                        "ind_ii_iii_cbm": row_["ind_ii_iii_cbm"],
                        "nsi_i_iii_cbm": row_["nsi_i_iii_cbm"],
                        "ass_iv_cbm": row_["ass_iv_cbm"],
                    }
                )

        microtec_df = pd.DataFrame(all_rows)

        # ---------- Merge mit PDF-Daten (unverändert) ----------
        df_prod["unterkategorie"] = df_prod["unterkategorie"].apply(normalize_dimension).apply(unify_dimension)

        merged_rows = []
        for _, pdfrow in df_prod.iterrows():
            ukey = pdfrow["unique_key"]
            ukat = pdfrow["unterkategorie"]
            row_dict = pdfrow.to_dict()

            if ukey is not None:
                match = microtec_df.loc[
                    (microtec_df["unique_key"] == ukey)
                    & (microtec_df["unterkategorie"] == ukat)
                ]
            else:
                match = pd.DataFrame()

            if not match.empty:
                rowM = match.iloc[0]
                row_dict.update(rowM.to_dict())
            else:
                # fehlende Werte mit 0 auffüllen
                for col in [
                    "Brutto_Volumen",
                    "waste_cbm",
                    "Netto_Volumen",
                    "Brutto_Ausbeute",
                    "Netto_Ausbeute",
                    "Vol_Eingang_m3",
                    "Brutto_Ausschuss",
                    "ce_cbm",
                    "kh_i_iii_cbm",
                    "sf_i_iii_cbm",
                    "sf_i_iiii_cbm",
                    "si_0_iv_cbm",
                    "si_i_ii_cbm",
                    "ind_ii_iii_cbm",
                    "nsi_i_iii_cbm",
                    "ass_iv_cbm",
                ]:
                    row_dict[col] = 0
            merged_rows.append(row_dict)

        merged_df = pd.DataFrame(merged_rows)
        st.session_state["merged_df"] = merged_df

        # ---------- Aggregation ----------
        df_agg = st.session_state["merged_df"].copy()
        numeric_cols = [
            "stämme",
            "vol_eingang",
            "durchschn_stammlänge",
            "teile",
            "Brutto_Volumen",
            "Brutto_Ausschuss",
            "Netto_Volumen",
            "Vol_Eingang_m3",
            "Brutto_Ausbeute",
            "Netto_Ausbeute",
            "ce_cbm",
            "kh_i_iii_cbm",
            "sf_i_iii_cbm",
            "sf_i_iiii_cbm",
            "si_0_iv_cbm",
            "si_i_ii_cbm",
            "ind_ii_iii_cbm",
            "nsi_i_iii_cbm",
            "ass_iv_cbm",
            "waste_cbm",
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
            "waste_cbm": "sum",
        }

        grouped = df_agg.groupby(
            ["unique_key", "auftrag", "unterkategorie"], as_index=False
        ).agg(agg_dict)

        # Ausschuss-% korrigieren
        grouped["Brutto_Ausschuss"] = grouped.apply(
            lambda r: round(
                100 * (r["waste_cbm"] / r["Brutto_Volumen"]) if r["Brutto_Volumen"] > 0 else 0,
                3,
            ),
            axis=1,
        )

        grouped["Netto_Volumen"] = grouped["Brutto_Volumen"] - grouped["waste_cbm"]

        # ---------- KORRIGIERT: Ausbeute erneut berechnen ----------
        def compute_yield_row(row, colname):
            return compute_yield(row[colname], row["Vol_Eingang_m3"])

        grouped["Brutto_Ausbeute"] = grouped.apply(
            lambda r: round(compute_yield_row(r, "Brutto_Volumen"), 3), axis=1
        )
        grouped["Netto_Ausbeute"] = grouped.apply(
            lambda r: round(compute_yield_row(r, "Netto_Volumen"), 3), axis=1
        )

        # ---------------------------------------------------------------------
        # SF-Spalten zusammenfassen
        grouped["sf_cbm"] = (
            grouped["kh_i_iii_cbm"] + grouped["sf_i_iii_cbm"] + grouped["sf_i_iiii_cbm"]
        )
        grouped.drop(
            ["kh_i_iii_cbm", "sf_i_iii_cbm", "sf_i_iiii_cbm"], axis=1, inplace=True
        )

        # SI-Spalten zusammenfassen
        grouped["si_cbm"] = grouped["si_0_iv_cbm"] + grouped["si_i_ii_cbm"]
        grouped.drop(["si_0_iv_cbm", "si_i_ii_cbm"], axis=1, inplace=True)

        final_cols = [
            "unique_key",
            "auftrag",
            "unterkategorie",
            "stämme",
            "vol_eingang",
            "durchschn_stammlänge",
            "teile",
            "Brutto_Volumen",
            "Brutto_Ausschuss",
            "Netto_Volumen",
            "Brutto_Ausbeute",
            "Netto_Ausbeute",
            "ce_cbm",
            "sf_cbm",
            "si_cbm",
            "ind_ii_iii_cbm",
            "nsi_i_iii_cbm",
            "ass_iv_cbm",
            "waste_cbm",
        ]
        for col in final_cols:
            if col not in grouped.columns:
                grouped[col] = 0

        grouped = grouped[final_cols]

        # ---------- Spalten-Mapping ----------
        rename_map = {
            "auftrag": "Auftrag",
            "unterkategorie": "Dimension",
            "stämme": "Stämme",
            "vol_eingang": "Volumen_Eingang",
            "durchschn_stammlänge": "Durchschn_Stämme",
            "teile": "Teile",
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
            "waste_cbm": "Ausschuss",
        }
        grouped.rename(columns=rename_map, inplace=True)

        # ---------- Durchmesser & Laufzeit (wie gehabt) ----------
        grouped["Durchmesser"] = 0.0
        grouped["Laufzeit_Minuten"] = 0.0

        for ukey, grp in grouped.groupby("unique_key"):
            idx = grp.index[0]
            vol_in = grp.loc[idx, "Volumen_Eingang"]
            durschn = grp.loc[idx, "Durchschn_Stämme"]
            stamme = grp.loc[idx, "Stämme"]
            try:
                diameter = round(math.sqrt(vol_in / (math.pi * durschn * stamme)) * 20000, 2)
            except Exception:
                diameter = 0
            grouped.loc[idx, "Durchmesser"] = diameter

            if ukey in orders_time:
                rt_start, rt_end = orders_time[ukey]["runtime"]
                runtime_minutes = (rt_end - rt_start).total_seconds() / 60
            else:
                runtime_minutes = 0
            grouped.loc[idx, "Laufzeit_Minuten"] = round(runtime_minutes, 2)

        grouped.drop("unique_key", axis=1, inplace=True)

        final_output_cols = [
            "Auftrag",
            "Dimension",
            "Stämme",
            "Volumen_Eingang",
            "Durchschn_Stämme",
            "Teile",
            "Durchmesser",
            "Laufzeit_Minuten",
            "Brutto_Volumen",
            "Brutto_Ausschuss",
            "Netto_Volumen",
            "Brutto_Ausbeute",
            "Netto_Ausbeute",
            "CE",
            "SF",
            "SI",
            "IND",
            "NSI",
            "Q_V",
            "Ausschuss",
        ]
        grouped = grouped[final_output_cols]

        # ---------- Ausgabe ----------
        st.subheader("Aggregiertes Ergebnis")
        st.dataframe(grouped)

        xlsx_data = to_excel_bytes(grouped)
        st.download_button(
            label="Download Aggregiertes Ergebnis",
            data=xlsx_data,
            file_name=f"Ausbeuteanalyse_{default_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

if __name__ == "__main__":
    main_app()
