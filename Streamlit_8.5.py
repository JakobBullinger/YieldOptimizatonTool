import streamlit as st
import pandas as pd
import pdfplumber
import re
from datetime import datetime, date, time
import io

###############################################################################
# 1) Synonym-Dictionary & Hilfsfunktionen
###############################################################################
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

###############################################################################
# 2) CSV-Funktionen (MicroTec)
###############################################################################
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

###############################################################################
# 3) Klassifizierung
###############################################################################
HW_DIMENSIONS = {
    "43x95","43x138","43x125","43x141","47x92","47x113","47x135","47x153","47x154",
    "47x175","47x198","47x220","47x221","51x104","36x134","48x117","48x128","48x144",
    "48x164","69x122","75x148","75x152","46x88"
}
SW_DIMENSIONS = {
    "17x75","17x78","17x100","17x98","23x103","26x146","28x131","28x149","35x155"
}
KH_DIMENSIONS = {
    "56x76","58x78","60x80","60x90","63x80","65x80","68x98","70x75","75x90","75x95",
    "75x98","76x76","76x96","78x78","78x90","78x100","79x100","93x93","96x116","98x98","33x90"
}

def classify_dimension(dim):
    dim_lower = dim.lower().strip()
    if dim_lower in (d.lower() for d in HW_DIMENSIONS):
        return "HW"
    elif dim_lower in (d.lower() for d in SW_DIMENSIONS):
        return "SW"
    elif dim_lower in (d.lower() for d in KH_DIMENSIONS):
        return "KH"
    else:
        return "Unknown"

###############################################################################
# 4) PDF Parsing
###############################################################################
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

###############################################################################
# NEU: Hilfsfunktion, um Aggregation durchzuführen, 
#      aber das Layout (Hauptzeile vs. Unterkategorien) beizubehalten
###############################################################################
def aggregate_in_original_format(df_prod):
    """
    Aggregiert Zeilen pro Auftrag + unterkategorie und gibt ein DataFrame
    zurück, das wieder eine Hauptzeile (unterkategorie=="") und
    darunter Sub-Zeilen für jede Unterkategorie hat – allerdings aggregiert.
    Wir gehen davon aus, dass df_prod das Format hat:
      - auftrag
      - unterkategorie ("" = Hauptzeile, != "" = Subzeile)
      - stämme, vol_eingang, durchschn_stammlänge, teile, vol_ausgang (o.ä.)

    Du kannst das Aggregationsdict nach Bedarf erweitern!
    """
    # 1) Aufteilen in Hauptzeilen vs. Unterzeilen
    main_df = df_prod[df_prod["unterkategorie"] == ""].copy()
    sub_df  = df_prod[df_prod["unterkategorie"] != ""].copy()

    # 2) Aggregation für Hauptzeilen
    main_agg = (
        main_df
        .groupby("auftrag", as_index=False)
        .agg({
            "stämme": "sum",
            "vol_eingang": "sum",
            "durchschn_stammlänge": "mean",  # oder "max", je nach Bedarf
            "teile": "sum",
            "vol_ausgang": "sum"
        })
    )

    # 3) Aggregation für Unterzeilen
    sub_agg = (
        sub_df
        .groupby(["auftrag", "unterkategorie"], as_index=False)
        .agg({
            "stämme": "sum",
            "vol_eingang": "sum",
            "durchschn_stammlänge": "mean",
            "teile": "sum",
            "vol_ausgang": "sum"
        })
    )

    # 4) Zusammenbauen im originalen PDF-Layout
    final_rows = []
    for _, mainrow in main_agg.iterrows():
        auftrag = mainrow["auftrag"]
        # Hauptzeile
        row_dict = {
            "auftrag": auftrag,
            "unterkategorie": "",
            "stämme": mainrow["stämme"],
            "vol_eingang": mainrow["vol_eingang"],
            "durchschn_stammlänge": mainrow["durchschn_stammlänge"],
            "teile": mainrow["teile"],
            "vol_ausgang": mainrow["vol_ausgang"],
        }
        final_rows.append(row_dict)

        # Subzeilen
        sub_of_order = sub_agg[sub_agg["auftrag"] == auftrag].copy()
        # Falls du die Subkategorien sortieren willst:
        # sub_of_order = sub_of_order.sort_values("unterkategorie")
        for _, subrow in sub_of_order.iterrows():
            sub_dict = {
                "auftrag": auftrag,
                "unterkategorie": subrow["unterkategorie"],
                # Wenn du willst, trägst du dieselben aggregierten Werte ein,
                # oder lässt manche Felder leer:
                "stämme": "",
                "vol_eingang": "",
                "durchschn_stammlänge": "",
                "teile": subrow["teile"],
                "vol_ausgang": subrow["vol_ausgang"]
            }
            final_rows.append(sub_dict)

    final_df = pd.DataFrame(final_rows)
    return final_df

###############################################################################
# 5) Streamlit-App
###############################################################################
def main_app():
    st.title("Produktivitäts- & MicroTec Auswertung (mit Aggregation im PDF-Layout)")

    ###########################################################################
    # 1. PDF hochladen & parsen
    ###########################################################################
    st.markdown("### 1. Produktivitätsbericht (PDF) hochladen")
    pdf_file = st.file_uploader("PDF des Produktivitätsberichts hochladen", type=["pdf"])
    
    orders_from_pdf = {}
    auftrag_infos = {}
    df_prod = None

    if pdf_file is not None:
        with st.spinner("PDF wird geparst..."):
            df_prod = extract_table_with_suborders_clean(pdf_file)
        if df_prod is not None:
            st.subheader("Geparster Produktivitätsbericht")
            st.dataframe(df_prod)
            
            # Typkonvertierung
            numeric_cols = ['stämme', 'vol_eingang', 'durchschn_stammlänge', 'teile', 'vol_ausgang']
            for col in numeric_cols:
                df_prod[col] = pd.to_numeric(
                    df_prod[col].astype(str).str.replace(',', '.').str.strip(), 
                    errors='coerce'
                )

            # PDF-Report als Excel-Download
            pdf_xlsx = to_excel_bytes(df_prod)
            st.download_button(
                label="Produktivitätsbericht als Excel herunterladen",
                data=pdf_xlsx,
                file_name="Produktivitätsbericht.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Hauptzeilen (unterkategorie == "") => unique_key pro Auftrag
            df_main = df_prod[df_prod["unterkategorie"] == ""].reset_index(drop=True)
            df_main["pdf_line_id"] = df_main.index

            # Alle Zeilen kriegen Spalte "unique_key".
            df_prod["unique_key"] = ""

            for i, row_ in df_main.iterrows():
                order = row_["auftrag"]
                pdf_line_id = row_["pdf_line_id"]
                unique_key = f"{order}_line{pdf_line_id}"

                # Setze 'unique_key' in df_prod für alle Zeilen mit diesem "auftrag"
                mask = (df_prod["auftrag"] == order)
                df_prod.loc[mask, "unique_key"] = unique_key

                # Ermittle vorhandene Unterkategorien
                group = df_prod.loc[mask]
                dims = group["unterkategorie"].unique().tolist()
                dims = [d for d in dims if d]  
                dims = [unify_dimension(normalize_dimension(d)) for d in dims]

                # Vol_Eingang aus Hauptzeile
                main_line = group[group["unterkategorie"] == ""]
                if not main_line.empty:
                    vol_in_str = main_line.iloc[0]["vol_eingang"]
                    vol_in = parse_vol_eingang(vol_in_str)
                else:
                    vol_in = 0.0

                orders_from_pdf[unique_key] = dims
                auftrag_infos[unique_key] = {"vol_eingang": vol_in}

    ###########################################################################
    # 2. CSV hochladen (MicroTec)
    ###########################################################################
    st.markdown("### 2. MicroTec CSV hochladen")
    csv_file = st.file_uploader("CSV der MicroTec Daten hochladen", type=["csv"])
    df_microtec = None
    if csv_file is not None:
        with st.spinner("MicroTec Daten werden geladen..."):
            df_microtec = load_and_prepare_data(csv_file)
        if df_microtec is not None:
            st.subheader("Vorschau MicroTec Daten")
            st.dataframe(df_microtec.head())
    
    if not orders_from_pdf:
        st.info("Bitte zuerst den Produktivitätsbericht hochladen und parsen.")
        st.stop()
    if df_microtec is None:
        st.info("Bitte auch die MicroTec CSV hochladen.")
        st.stop()

    ###########################################################################
    # 3. Auftragszeitfenster festlegen
    ###########################################################################
    default_date = df_microtec['Datetime'].iloc[0].date()
    st.markdown("### 3. Auftragszeitfenster festlegen (nur Uhrzeit)")
    orders_final = {}
    for ukey in orders_from_pdf.keys():
        st.markdown(f"#### Auftrag: {ukey}")
        col1, col2 = st.columns(2)
        start_hour = col1.number_input(f"Start Stunde {ukey}", 0, 23, 6, 1)
        start_minute = col1.number_input(f"Start Minute {ukey}", 0, 59, 0, 1)
        end_hour = col2.number_input(f"Ende Stunde {ukey}", 0, 23, 12, 1)
        end_minute = col2.number_input(f"Ende Minute {ukey}", 0, 59, 0, 1)

        start_time = time(start_hour, start_minute)
        end_time = time(end_hour, end_minute)
        start_dt = datetime.combine(default_date, start_time)
        end_dt = datetime.combine(default_date, end_time)

        orders_final[ukey] = {
            "time_window": (start_dt, end_dt),
            "dimensions": orders_from_pdf[ukey]
        }

    def compute_yield(volume, vol_in_liters):
        if vol_in_liters == 0:
            return 0
        return (volume / (vol_in_liters / 1000.0)) * 100

    final_rows = []

    ###########################################################################
    # 4. Button "Auswerten" => MicroTec & Zusammenführen
    ###########################################################################
    if st.button("Auswerten"):
        st.markdown("### 4. Ergebnisse")

        # 4a) MicroTec-Auswertung
        for ukey, params in orders_final.items():
            (start_dt, end_dt) = params["time_window"]
            dims = params["dimensions"]
            df_order = filter_data_for_order(df_microtec, start_dt, end_dt, dims)
            result = summarize_cbm_and_waste(df_order)

            vol_in = auftrag_infos[ukey]["vol_eingang"]
            
            for idx, row_ in result.iterrows():
                dim = row_["Dimension"]
                total_cbm = row_["total_cbm"]
                waste_cbm = row_["waste_cbm"]
                waste_percent = row_["waste_percent"]
                
                netto_vol = total_cbm - waste_cbm
                brutto_ausb = compute_yield(total_cbm, vol_in)
                netto_ausb = compute_yield(netto_vol, vol_in)
                warentyp = classify_dimension(dim)

                final_rows.append({
                    "unique_key": ukey,
                    "Dimension": dim,
                    "Warentyp": warentyp,
                    "Brutto_Volumen": round(total_cbm, 3),
                    "Brutto-Ausschuss": round(waste_percent, 2),
                    "Netto_Volumen": round(netto_vol, 3),
                    "Brutto_Ausbeute (%)": round(brutto_ausb, 2),
                    "Netto_Ausbeute (%)": round(netto_ausb, 2),
                    "Vol_Eingang (in m³)": round(vol_in / 1000, 3),
                })
        
        final_df = pd.DataFrame(final_rows)
        st.subheader("Finale Ausbeute-Tabelle (nur MicroTec-Berechnung)")
        st.dataframe(final_df)

        # Download rohes MicroTec-Ergebnis
        microtec_xlsx = to_excel_bytes(final_df)
        st.download_button(
            label="Download MicroTec-Auswertung als Excel",
            data=microtec_xlsx,
            file_name="Ausbeute_MicroTec.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        #######################################################################
        # 5. Zusammenführen (manueller Join) via (unique_key, Dimension)
        #######################################################################
        st.markdown("### 5. Zusammenführen (PDF + MicroTec)")

        # Umbenennen => "unterkategorie"
        final_df.rename(columns={"Dimension": "unterkategorie"}, inplace=True)

        # Dictionary: key = (unique_key, unterkategorie)
        microtec_dict = {}
        for _, row_ in final_df.iterrows():
            k = (row_["unique_key"], row_["unterkategorie"])
            microtec_dict[k] = {
                "Warentyp": row_["Warentyp"],
                "Brutto_Volumen": row_["Brutto_Volumen"],
                "Brutto-Ausschuss": row_["Brutto-Ausschuss"],
                "Netto_Volumen": row_["Netto_Volumen"],
                "Brutto_Ausbeute": row_["Brutto_Ausbeute (%)"],
                "Netto_Ausbeute": row_["Netto_Ausbeute (%)"],
                "Vol_Eingang_m3": row_["Vol_Eingang (in m³)"],
            }

        # PDF "unterkategorie" normalisieren
        df_prod["unterkategorie"] = (
            df_prod["unterkategorie"]
            .apply(normalize_dimension)
            .apply(unify_dimension)
        )

        merged_rows = []
        for idx, pdf_row in df_prod.iterrows():
            k = (pdf_row["unique_key"], pdf_row["unterkategorie"])
            row_dict = pdf_row.to_dict()
            
            if k in microtec_dict:
                row_dict.update(microtec_dict[k])
            else:
                # Keine MicroTec-Daten
                row_dict["Warentyp"] = ""
                row_dict["Brutto_Volumen"] = ""
                row_dict["Brutto-Ausschuss"] = ""
                row_dict["Netto_Volumen"] = ""
                row_dict["Brutto_Ausbeute"] = ""
                row_dict["Netto_Ausbeute"] = ""
                row_dict["Vol_Eingang_m3"] = ""

            merged_rows.append(row_dict)
        
        merged_df = pd.DataFrame(merged_rows)

        # Spaltenreihenfolge
        final_cols = [
            "auftrag", "unterkategorie", "stämme", "vol_eingang", "durchschn_stammlänge",
            "teile", "vol_ausgang", "Warentyp", "Brutto_Volumen", "Brutto-Ausschuss",
            "Netto_Volumen", "Brutto_Ausbeute", "Netto_Ausbeute", "Vol_Eingang_m3",
            "unique_key"
        ]
        for c in final_cols:
            if c not in merged_df.columns:
                merged_df[c] = ""
        merged_df = merged_df[final_cols]
        merged_df.fillna("", inplace=True)

        st.subheader("Manuell zusammengeführte Tabelle (PDF + MicroTec)")
        st.dataframe(merged_df)

        merged_xlsx = to_excel_bytes(merged_df)
        st.download_button(
            label="Download Zusammengeführte Excel",
            data=merged_xlsx,
            file_name="Produktivitaet_MicroTec_merged.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        #######################################################################
        # 6. (NEU) Aggregieren & Layout "wie im PDF"
        #######################################################################
        st.markdown("### 6. Aggregation + PDF-Layout")
        if st.button("Aggregation im PDF-Layout"):
            # Zuerst numerische Spalten in float umwandeln
            numeric_for_agg = [
                "stämme", "vol_eingang", "durchschn_stammlänge",
                "teile", "vol_ausgang",
                "Brutto_Volumen", "Netto_Volumen", "Brutto_Ausbeute", 
                "Netto_Ausbeute", "Vol_Eingang_m3"
            ]
            for c in numeric_for_agg:
                merged_df[c] = pd.to_numeric(merged_df[c], errors="coerce")

            # Jetzt rufen wir unser aggregator-Tool auf
            final_agg_df = aggregate_in_original_format(merged_df)
            st.subheader("Aggregiert (PDF-Layout)")
            st.dataframe(final_agg_df)

            # Download
            agg_xlsx = to_excel_bytes(final_agg_df)
            st.download_button(
                label="Download Aggregierte Tabelle",
                data=agg_xlsx,
                file_name="Aggregierte_PDF_Layout.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main_app()
