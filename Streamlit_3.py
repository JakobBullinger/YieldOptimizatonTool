import streamlit as st
import pandas as pd
import pdfplumber
import re
from datetime import datetime, time, timedelta

# --- Funktionen für die MicroTec-Daten ---
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

def load_and_prepare_data(filepath):
    """
    Liest die CSV-Datei (ohne Header) ein und bereitet die Daten vor:
      - Liest die Datei mit header=None und sep=';'
      - Benennt die relevanten Spalten anhand der Indexposition:
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
      - waste_cbm: Summe der CBM, bei denen Classification == 'Waste'
      - waste_percent: Prozentualer Ausschuss (waste_cbm / total_cbm * 100)
    """
    if df.empty:
        return pd.DataFrame(columns=['Dimension', 'total_cbm', 'waste_cbm', 'waste_percent'])
    
    grouped = df.groupby('Dimension').agg(
        total_cbm=('CBM', 'sum'),
        waste_cbm=('CBM', lambda x: x[df.loc[x.index, 'Classification'] == 'Waste'].sum())
    ).reset_index()
    
    grouped['waste_percent'] = (grouped['waste_cbm'] / grouped['total_cbm'] * 100).round(2)
    grouped['total_cbm'] = grouped['total_cbm'].round(3)
    grouped['waste_cbm'] = grouped['waste_cbm'].round(3)
    return grouped

# --- Funktionen zum Parsen des Produktivitätsberichts (PDF) ---
def extract_table_with_suborders_clean(file_path, start_keyword="Auftrag"):
    # PDF komplett einlesen
    with pdfplumber.open(file_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    
    # Ab Startpunkt (z.B. "Auftrag")
    start_index = full_text.find(start_keyword)
    if start_index == -1:
        raise ValueError(f"Startkeyword '{start_keyword}' nicht gefunden!")
    table_text = full_text[start_index:]
    
    # Alle Zeilen sammeln und leere Zeilen entfernen
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    
    # Angepasster Regex für Haupt- und Unterzeilen:
    # Der Regex für die Hauptzeile erfasst ab dem Start (Auftragsnummer etc.) bis zu den fünf numerischen Feldern.
    # Zusätzlich wird in der optionalen Gruppe "extra" alles erfasst, was danach steht.
    main_row_pattern = re.compile(
        r"^(?P<auftrag>\d{5}\s*-\s*.*?)(?=\s+[\d.,]+\s+)"  # erfasst den Auftragstext (non-gierig) bis zu einer Folge aus Ziffern
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
    
    # Verbesserter Merge-Algorithmus (bleibt unverändert):
    merged_lines = []
    buffer = ""
    for line in lines:
        # Falls Zeile als Unterzeile erkannt wird, ...
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

        # Prüfen, ob Zeile mit einem Hauptzeilen-Kandidaten beginnt (5 Ziffern und Bindestrich)
        if re.match(r'^\d{5}\s*-\s*', line):
            if buffer:
                if "Auftrag" in buffer:
                    m = re.search(r'\d{5}\s*-\s*', buffer)
                    if m:
                        buffer = buffer[m.start():]
                merged_lines.append(buffer)
            buffer = line
        else:
            # Andernfalls als Fortsetzung der aktuellen Hauptzeile interpretieren
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
    
    # Debug: Ausgabe der zusammengeführten Zeilen
    for ml in merged_lines:
        print("MERGED:", repr(ml))
    
    # Jetzt parsen: Zuerst Hauptzeilen, danach Unterzeilen zuordnen
    result_rows = []
    current_main_order = None
    for line in merged_lines:
        line = line.strip()
        if not line:
            continue
        
        main_match = main_row_pattern.match(line)
        if main_match:
            main_dict = main_match.groupdict()
            # Falls in der "extra"-Gruppe noch Text steht, prüfen wir, ob das letzte Token nicht rein numerisch ist.
            auftrag = main_dict['auftrag'].strip()
            extra = main_dict.get('extra', '').strip()
            if extra:
                tokens = extra.split()
                # Wenn das letzte Token nicht nur aus Ziffern (mit Komma/Punkt) besteht, wird es zum Auftragstext hinzugefügt.
                if tokens and not re.fullmatch(r'[\d.,]+', tokens[-1]):
                    auftrag += " " + tokens[-1]
            current_main_order = main_dict
            current_main_order['auftrag'] = auftrag
            row = {
                'auftrag': current_main_order['auftrag'],
                'unterkategorie': "",  # leer, weil Hauptauftrag
                'stämme': main_dict['stämme'],
                'vol_eingang': main_dict['vol_eingang'],
                'durchschn_stammlänge': main_dict['durchschn_stammlänge'],
                'teile': main_dict['teile'],
                'vol_ausgang': main_dict['vol_ausgang'],
            }
            result_rows.append(row)
        else:
            # Versuch als Unterzeile zu parsen, falls es aktuell einen Hauptauftrag gibt
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

# --- Streamlit-App ---
def main_app():
    st.title("Produktivitäts- & MicroTec Auswertung")
    
    st.markdown("### 1. Produktivitätsbericht (PDF) hochladen")
    pdf_file = st.file_uploader("PDF des Produktivitätsberichts hochladen", type=["pdf"])
    
    orders_from_pdf = {}
    if pdf_file is not None:
        with st.spinner("PDF wird geparst..."):
            df_prod = extract_table_with_suborders_clean(pdf_file)
        if df_prod is not None:
            st.subheader("Geparster Produktivitätsbericht")
            st.dataframe(df_prod)
            # Erhalte die Aufträge in der Reihenfolge, in der sie im PDF erscheinen
            for order in df_prod["auftrag"].unique():
                group = df_prod[df_prod["auftrag"] == order]
                dims = group["unterkategorie"].unique().tolist()
                dims = [d for d in dims if d]  # leere Strings entfernen
                # Vereinheitliche die Dimensionen
                dims = [unify_dimension(normalize_dimension(d)) for d in dims]
                orders_from_pdf[order] = list(dict.fromkeys(dims))  # Erhalte die Reihenfolge, wie sie erscheinen
            
            st.markdown("**Automatisch ermittelte Dimensionen pro Auftrag:**")
            for order, dims in orders_from_pdf.items():
                st.write(f"{order}: {dims}")
    
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
    
    # Als Basisdatum nutzen wir das Datum des ersten Datensatzes, aber der Nutzer wählt nur die Uhrzeiten.
    default_date = df_microtec['Datetime'].iloc[0].date()
    
    st.markdown("### 3. Auftragszeitfenster festlegen (nur Uhrzeit)")
    orders_final = {}
    # Für jeden Auftrag (in der Reihenfolge aus dem PDF)
    for order in orders_from_pdf.keys():
        st.markdown(f"#### Auftrag: {order}")
        col1, col2 = st.columns(2)

        start_hour = col1.number_input(
            f"Start Stunde {order}", 
            min_value=0, max_value=23, 
            value=6, step=1, key=f"start_hour_{order}"
        )
        
        end_hour = col1.number_input(
            f"Ende Stunde {order}", 
            min_value=0, max_value=23, 
            value=12, step=1, key=f"end_hour_{order}"
        )

        start_minute = col2.number_input(
            f"Start Minute {order}", 
            min_value=0, max_value=59, 
            value=0, step=1, key=f"start_minute_{order}"
        )
        start_time = time(start_hour, start_minute)

        end_minute = col2.number_input(
            f"Ende Minute {order}", 
            min_value=0, max_value=59, 
            value=0, step=1, key=f"end_minute_{order}"
        )
        end_time = time(end_hour, end_minute)


        ### Minütliches Dropdown menü
        # start_time = col1.time_input(
        #     f"Startzeit {order}",
        #     value=time(6,0),
        #     step=timedelta(minutes=1),   
        #     key=f"start_time_{order}"
        # )
        # end_time = col2.time_input(
        #     f"Endzeit {order}",
        #     value=time(12,0),
        #     step=timedelta(minutes=1),
        #     key=f"end_time_{order}"
        # )

        # start_time = col1.time_input(f"Startzeit {order}", value=time(6,0), key=f"start_time_{order}")
        # end_time = col2.time_input(f"Endzeit {order}", value=time(12,0), key=f"end_time_{order}")
        # Kombiniere die Uhrzeit mit dem automatisch ermittelten Datum
        start_dt = datetime.combine(default_date, start_time)
        end_dt = datetime.combine(default_date, end_time)
        
        orders_final[order] = {
            "time_window": (start_dt, end_dt),
            "dimensions": orders_from_pdf[order]  # automatisch aus PDF
        }
    
    if st.button("Auswerten"):
        st.markdown("### 4. Ergebnisse")
        for order, params in orders_final.items():
            start_dt, end_dt = params["time_window"]
            dims = params["dimensions"]
            df_order = filter_data_for_order(df_microtec, start_dt, end_dt, dims)
            result = summarize_cbm_and_waste(df_order)
            st.markdown(f"**{order}** (Zeitraum: {start_dt.time()} bis {end_dt.time()})")
            st.dataframe(result)
            st.markdown("---")

if __name__ == "__main__":
    main_app()
