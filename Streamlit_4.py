import streamlit as st
import pandas as pd
import pdfplumber
import re
from datetime import datetime, date, time, timedelta
import io
import xlsxwriter


# --- Dimension-Mapping (MicroTec) ---
synonyms = {
    "17x98": "17x100",
    "17x95": "17x100",
    "23x100": "23x103",
    "47x212": "47x221"
}

def unify_dimension(dim_str):
    """Wendet das Mapping an."""
    return synonyms.get(dim_str, dim_str)

def normalize_dimension(dim_str):
    """Normalisiert Strings wie '17.0 x 100.0 mm' zu '17x100'."""
    if pd.isna(dim_str):
        return "Unknown"
    dim_str = (str(dim_str)
               .lower()
               .replace('mm', '')
               .replace(',', '.')
               .replace(' ', '')
               .replace('.0', ''))
    return dim_str

# --- Funktionen für die MicroTec CSV-Daten ---
def load_and_prepare_data(filepath):
    """
    Liest die CSV-Datei (ohne Header) ein und bereitet die Daten vor:
      - Liest mit header=None und sep=';'
      - Benennt relevante Spalten anhand ihrer Indexposition:
          Index 1  -> 'Year'
          Index 2  -> 'Month'
          Index 3  -> 'Day'
          Index 4  -> 'Hour'
          Index 5  -> 'Minute'
          Index 15 -> 'Dimension'
          Index 21 -> 'Classification'
          Index 26 -> 'CBM'
      - Erstellt eine Datetime-Spalte
      - Normalisiert und vereinheitlicht die Dimensionen
    """
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
    """
    Filtert das DataFrame nach:
      - Zeitfenster [start_dt, end_dt)
      - Nur den angegebenen Dimensionen
    """
    mask = (
        (df['Datetime'] >= start_dt) &
        (df['Datetime'] < end_dt) &
        (df['Dimension'].isin(dimensions))
    )
    return df.loc[mask].copy()

def summarize_cbm_and_waste(df):
    """
    Gruppiert das DataFrame nach 'Dimension' und berechnet:
      - total_cbm: Summe der CBM
      - waste_cbm: Summe der CBM, wo Classification == 'Waste'
      - waste_percent: (waste_cbm/total_cbm * 100)
    """
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

# --- Definition der Klassifizierungslisten und Schwund-Faktoren ---
HW_DIMENSIONS = {
    "43x95","43x138","43x125","43x141","47x92","47x113","47x135","47x153","47x154",
    "47x175","47x198","47x220","47x221","51x104","36x134","48x117","48x128","48x144",
    "48x164","69x122","75x148","75x152"
}
SW_DIMENSIONS = {
    "17x75","17x78","17x100","17x98","23x103","26x146","28x131","28x149","35x155"
}
KH_DIMENSIONS = {
    "56x76","58x78","60x80","60x90","63x80","65x80","68x98","70x75","75x90","75x95",
    "75x98","76x76","76x96","78x78","78x90","78x100","79x100","93x93","96x116","98x98","33x90"
}
SCHWUND_FACTORS = {
    "HW": 0.11,  # 11%
    "SW": 0.09,  # 9%
    "KH": 0.00
}

def classify_dimension(dim):
    """
    Gibt 'HW', 'SW', 'KH' oder 'Unknown' zurück, basierend auf den Dimensionen.
    """
    dim_lower = dim.lower().strip()
    if dim_lower in (d.lower() for d in HW_DIMENSIONS):
        return "HW"
    elif dim_lower in (d.lower() for d in SW_DIMENSIONS):
        return "SW"
    elif dim_lower in (d.lower() for d in KH_DIMENSIONS):
        return "KH"
    else:
        return "Unknown"

# --- Funktionen zum Parsen des Produktivitätsberichts (PDF) ---
def extract_table_with_suborders_clean(file_input, start_keyword="Auftrag"):
    """
    Parst den PDF-Text des Produktivitätsberichts und liefert ein DataFrame mit:
      - 'auftrag' (z. B. "10421 - 3x47x135")
      - 'unterkategorie' (z. B. "17x95", "47x135")
      - sowie weiteren Spalten wie 'vol_eingang'
    """
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
    
    # Regex-Definitionen
    main_row_pattern = re.compile(
        r"^(?P<auftrag>\d{5}\s*-\s*.*?)(?=\s+[\d.,]+\s+)"
        r"\s+(?P<stämme>[\d.,]+)\s+"
        r"(?P<vol_eingang>[\d.,]+)\s+"
        r"(?P<durchschn_stammlänge>[\d.,]+)\s+"
        r"(?P<teile>[\d.,]+)\s+"
        r"(?P<vol_ausgang>[\d.,]+)"
        r"(?:\s+.*)?$"
    )
    sub_row_pattern = re.compile(
        r"^(?P<unterkategorie>\d+x\d+)"
    )
    
    merged_lines = []
    buffer = ""
    for line in lines:
        if sub_row_pattern.match(line):
            if buffer:
                merged_lines.append(buffer)
                buffer = ""
            merged_lines.append(line)
        else:
            if re.match(r'^\d{5}\s*-\s*', line):
                if buffer:
                    merged_lines.append(buffer)
                buffer = line
            else:
                if buffer:
                    buffer += " " + line
                else:
                    buffer = line
    if buffer:
        merged_lines.append(buffer)
    
    result_rows = []
    current_order = None
    for line in merged_lines:
        line = line.strip()
        if not line:
            continue
        main_match = main_row_pattern.match(line)
        if main_match:
            main_dict = main_match.groupdict()
            auftrag = main_dict['auftrag'].strip()
            current_order = auftrag
            row = {
                'auftrag': auftrag,
                'unterkategorie': "",  # Hauptzeile
                'stämme': main_dict['stämme'],
                'vol_eingang': main_dict['vol_eingang'],
                'durchschn_stammlänge': main_dict['durchschn_stammlänge'],
                'teile': main_dict['teile'],
                'vol_ausgang': main_dict['vol_ausgang'],
            }
            result_rows.append(row)
        else:
            sub_match = sub_row_pattern.match(line)
            if sub_match and current_order:
                dim = sub_match.group("unterkategorie")
                result_rows.append({
                    'auftrag': current_order,
                    'unterkategorie': dim,
                    'stämme': "",
                    'vol_eingang': "",
                    'durchschn_stammlänge': "",
                    'teile': "",
                    'vol_ausgang': ""
                })
    return pd.DataFrame(result_rows)

def parse_vol_eingang(vol_str):
    """
    Konvertiert einen String wie '437.498' in einen Float.
    Entfernt Tausendertrennzeichen und ersetzt Komma durch Punkt.
    """
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

# --- Streamlit-App ---
def main_app():
    st.title("Produktivitäts- & MicroTec Auswertung")
    
    st.markdown("### 1. Produktivitätsbericht (PDF) hochladen")
    pdf_file = st.file_uploader("PDF des Produktivitätsberichts hochladen", type=["pdf"])
    
    orders_from_pdf = {}
    auftrag_infos = {}  # Speichert vol_eingang pro Auftrag
    df_prod = None
    if pdf_file is not None:
        with st.spinner("PDF wird geparst..."):
            df_prod = extract_table_with_suborders_clean(pdf_file)
        if df_prod is not None:
            st.subheader("Geparster Produktivitätsbericht")
            st.dataframe(df_prod)
            # Für jeden Auftrag: Dimensionen extrahieren und vol_eingang ermitteln
            for order in df_prod["auftrag"].unique():
                group = df_prod[df_prod["auftrag"] == order]
                dims = group["unterkategorie"].unique().tolist()
                dims = [d for d in dims if d]  # leere Strings entfernen
                dims = [unify_dimension(normalize_dimension(d)) for d in dims]
                orders_from_pdf[order] = list(dict.fromkeys(dims))
                
                main_line = group[group["unterkategorie"] == ""]
                if not main_line.empty:
                    vol_in_str = main_line.iloc[0]["vol_eingang"]
                    vol_in = parse_vol_eingang(vol_in_str)
                else:
                    vol_in = 0.0
                auftrag_infos[order] = {"vol_eingang": vol_in}
            
            st.markdown("**Ermittelte Dimensionen & Vol_Eingang pro Auftrag:**")
            for order, dims in orders_from_pdf.items():
                st.write(f"{order}: {dims} | Vol_Eingang: {auftrag_infos[order]['vol_eingang']}")
    
    st.markdown("### 2. MicroTec CSV hochladen")
    csv_file = st.file_uploader("CSV der MicroTec Daten hochladen", type=["csv"])
    
    df_microtec = None
    if csv_file is not None:
        with st.spinner("MicroTec Daten werden geladen..."):
            df_microtec = load_and_prepare_data(csv_file)
        st.subheader("Vorschau MicroTec Daten")
        st.dataframe(df_microtec.head())
    
    if not orders_from_pdf:
        st.info("Bitte zuerst den Produktivitätsbericht hochladen und parsen.")
        st.stop()
    if df_microtec is None:
        st.info("Bitte auch die MicroTec CSV hochladen.")
        st.stop()
    
    # Basisdatum aus MicroTec-Daten (Datum spielt hier keine Rolle – nur Uhrzeit)
    default_date = df_microtec['Datetime'].iloc[0].date()
    
    st.markdown("### 3. Auftragszeitfenster festlegen (nur Uhrzeit)")
    orders_final = {}
    for order in orders_from_pdf.keys():
        st.markdown(f"#### Auftrag: {order}")
        col1, col2 = st.columns(2)
        start_hour = col1.number_input(f"Start Stunde {order}", min_value=0, max_value=23, value=6, step=1, key=f"start_hour_{order}")
        start_minute = col1.number_input(f"Start Minute {order}", min_value=0, max_value=59, value=0, step=1, key=f"start_minute_{order}")
        end_hour = col2.number_input(f"Ende Stunde {order}", min_value=0, max_value=23, value=12, step=1, key=f"end_hour_{order}")
        end_minute = col2.number_input(f"Ende Minute {order}", min_value=0, max_value=59, value=0, step=1, key=f"end_minute_{order}")
        start_time = time(start_hour, start_minute)
        end_time = time(end_hour, end_minute)
        start_dt = datetime.combine(default_date, start_time)
        end_dt = datetime.combine(default_date, end_time)
        orders_final[order] = {
            "time_window": (start_dt, end_dt),
            "dimensions": orders_from_pdf[order]
        }
    
    # --- Zusätzliche Berechnungen: Klassifizierung, Brutto-/Netto-Ausbeute ---
    # Funktionen zur Berechnung
    def compute_yield(volume, vol_eingang):
        return (volume / (vol_eingang/1000)) * 100 if vol_eingang != 0 else 0
    
    # Erstelle finalen Output (eine Zeile pro Auftrag und Dimension)
    final_rows = []
    if st.button("Auswerten"):
        st.markdown("### 4. Ergebnisse")
        # Für jeden Auftrag: Filter MicroTec-Daten, gruppiere nach Dimension
        for order, params in orders_final.items():
            start_dt, end_dt = params["time_window"]
            dims = params["dimensions"]
            df_order = filter_data_for_order(df_microtec, start_dt, end_dt, dims)
            result = summarize_cbm_and_waste(df_order)
            vol_eingang = auftrag_infos[order]["vol_eingang"]
            # Für jede Dimension in result
            for idx, row in result.iterrows():
                dimension = row["Dimension"]
                brutto_vol = row["total_cbm"]
                waste_vol = row["waste_cbm"]
                netto_vol = brutto_vol * (1 - (waste_vol / brutto_vol)) if brutto_vol != 0 else 0
                brutto_ausbeute = compute_yield(brutto_vol, vol_eingang)
                netto_ausbeute = compute_yield(netto_vol, vol_eingang)
                # Klassifikation anhand der Dimension
                warentyp = classify_dimension(dimension)
                final_rows.append({
                    "Auftrag": order,
                    "Dimension": dimension,
                    "Warentyp": warentyp,
                    "Brutto_Volumen": round(brutto_vol, 3),
                    "Brutto_Waste": round(waste_vol, 3),
                    "Netto_Volumen": round(netto_vol, 3),
                    "Brutto_Ausbeute (%)": round(brutto_ausbeute, 2),
                    "Netto_Ausbeute (%)": round(netto_ausbeute, 2),
                    "Vol_Eingang (in m³)": round(vol_eingang/1000, 3)
                })
        
        final_df = pd.DataFrame(final_rows)
        st.subheader("Finale Ausbeute-Tabelle")
        st.dataframe(final_df)
        
        # Excel-Export
        excel_data = to_excel_bytes(final_df)
        st.download_button(
            label="Download als Excel",
            data=excel_data,
            file_name="Ausbeute_Auswertung.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main_app()
