import pandas as pd
from datetime import datetime

def load_and_prepare_data(filepath):
    """
    Liest deine Excel-Datei ein und bereitet die Daten vor:
      - Spalten umbenennen (z.B. Column2 -> 'Year' usw.)
      - Datetime-Spalte erstellen
      - Dimension normalisieren
    Passe die Spaltennamen ggf. an deinen konkreten Fall an!
    """
    # Beispielhaftes Einlesen mit Header=0 (falls deine Datei Spaltennamen hat)
    df = pd.read_excel(filepath, header=0)
    
    # Beispiel: Spalten umbenennen
    df.rename(columns={
        "Column2": "Year",
        "Column3": "Month",
        "Column4": "Day",
        "Column5": "Hour",
        "Column6": "Minute",
        "Column16": "Dimension",
        "Column22": "Classification",
        "Column27": "CBM"
    }, inplace=True)
    
    # Erstelle eine Datetime-Spalte
    df['Datetime'] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute']])
    
    # Dimension normalisieren
    df['Dimension'] = df['Dimension'].apply(normalize_dimension)
    
    return df

def normalize_dimension(dim_str):
    """
    Wandelt einen String wie '17.0 x 100.0 mm' in eine einheitliche Form um, z.B. '17x100'.
    Passe diesen Teil ggf. an deine Datenqualität an.
    """
    if pd.isna(dim_str):
        return "Unknown"
    dim_str = (str(dim_str)
               .lower()
               .replace('mm','')
               .replace(',', '.')
               .replace(' ', '')
               .replace('.0',''))
    return dim_str

def filter_data_for_order(df, start_dt, end_dt, dimensions):
    """
    Filtert das DataFrame nach:
      1) Zeit: [start_dt, end_dt)
      2) Dimension: nur die gewünschten Dimensionen.
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
      - total_cbm: aufsummierte CBM
      - waste_cbm: aufsummierte CBM, wo Classification == 'Waste'
      - waste_percent: prozentualer Ausschuss (waste_cbm / total_cbm * 100)
    """
    if df.empty:
        # Falls der DataFrame leer ist, gib einen leeren Frame zurück
        return pd.DataFrame(columns=['Dimension', 'total_cbm', 'waste_cbm', 'waste_percent'])
    
    grouped = df.groupby('Dimension').agg(
        total_cbm=('CBM', 'sum'),
        waste_cbm=('CBM', lambda x: x[df.loc[x.index, 'Classification'] == 'Waste'].sum())
    ).reset_index()
    
    grouped['waste_percent'] = (grouped['waste_cbm'] / grouped['total_cbm'] * 100).round(2)
    grouped['total_cbm'] = grouped['total_cbm'].round(3)
    grouped['waste_cbm'] = grouped['waste_cbm'].round(3)
    
    return grouped

def main():
    """
    Hauptprogramm:
      1) Daten laden und vorbereiten
      2) Auftrags-Infos definieren (Start/Endzeit + Dimensionen)
      3) Filter und Zusammenfassung pro Auftrag
      4) Ausgabe
    """
    # 1. Daten laden
    filepath = r"C:\Users\jfxbu\OneDrive - Universitaet St.Gallen\Dokumente\St.Gallen\Praktika\Gelo\Produktivitätsbericht\MicroTec\2024_12_5_Formatted.xlsx"

    df = load_and_prepare_data(filepath)
    
    # 2. Definiere deine Aufträge:
    #    Für jeden Auftrag gibst du (start_dt, end_dt) und die zugehörigen Dimensionen an.
    #    Beispiel: Auftrag A => 06:00 - 11:25, Dimensionen ["17x95", "47x135"]
    
    orders = {
        "Auftrag A": {
            "time_window": (datetime(2024, 12, 5, 6, 0),
                            datetime(2024, 12, 5, 11, 23)),
            "dimensions": ["17x100", "47x135"]
        },
        "Auftrag B": {
            "time_window": (datetime(2024, 12, 5, 11, 23),
                            datetime(2024, 12, 5, 12, 0)),
            "dimensions": ["76x96"]
        },
        # Füge weitere Aufträge nach Bedarf hinzu ...
    }
    
    # 3. Filterung und Auswertung pro Auftrag
    for order_name, order_info in orders.items():
        start_dt, end_dt = order_info["time_window"]
        dims = order_info["dimensions"]
        
        df_order = filter_data_for_order(df, start_dt, end_dt, dims)
        result = summarize_cbm_and_waste(df_order)
        
        # 4. Ausgabe
        print(f"=== {order_name} ===")
        print(result)
        print()

if __name__ == "__main__":
    main()
