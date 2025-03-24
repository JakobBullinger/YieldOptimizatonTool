# import pandas as pd
# from datetime import datetime

# def load_and_prepare_data(filepath):
#     """
#     Lädt das Excel-File und bereitet die relevanten Spalten auf.
#     Falls das Excel bereits Spaltenüberschriften besitzt, wird header=0 genutzt.
#     """
#     # Falls deine Excel-Datei Überschriften hat, benutze header=0
#     df = pd.read_excel(filepath, header=0)  # header=0, falls Überschriften vorhanden
    
#     # Debug: Zeige alle Spaltennamen an, um sicherzustellen, dass wir die richtigen haben.
#     print("Spalten im DataFrame:", df.columns.tolist())
    
#     # Passe die Umbenennung an, falls die Spaltennamen oder ihre Position anders sind.
#     # Beispiel: Wenn die Excel-Datei bereits aussagekräftige Überschriften hat, musst du evtl. nicht umbenennen.
#     # Falls du jedoch umbenennen möchtest, passe die Keys hier an.
#     df.rename(columns={
#         'Jahr': 'Year',
#         'Monat': 'Month',
#         'Tag': 'Day',
#         'Stunde': 'Hour',
#         'Minute': 'Minute',
#         'Dimension': 'Dimension',          # falls bereits so, kannst du es auch weglassen
#         'Klassifizierung': 'Classification',
#         'CBM': 'CBM'
#     }, inplace=True)
    
#     # Überprüfe, ob die benötigten Spalten jetzt vorhanden sind:
#     required_columns = ['Year', 'Month', 'Day', 'Hour', 'Minute']
#     for col in required_columns:
#         if col not in df.columns:
#             raise KeyError(f"Erforderliche Spalte '{col}' fehlt im DataFrame!")
    
#     # Erzeuge einen Datetime-Index oder eine Datetime-Spalte
#     df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    
#     # Dimension säubern (z.B. '17.0 x 100.0 mm' -> '17x100')
#     df['Dimension'] = df['Dimension'].apply(normalize_dimension)
    
#     return df

# def normalize_dimension(dim_str):
#     """
#     Bereinigt den Dimensions-String.
#     Beispiel: '17.0 x 100.0 mm' -> '17x100'
#     """
#     if pd.isna(dim_str):
#         return "Unknown"
    
#     dim_str = str(dim_str).lower().replace('mm', '').replace(',', '.')
#     dim_str = dim_str.strip().replace(' ', '').replace('.0', '')
#     return dim_str

# def filter_data_for_order(df, start_dt, end_dt):
#     """
#     Filtert das DataFrame df auf die Zeilen, die in der Zeitspanne [start_dt, end_dt) liegen.
#     """
#     mask = (df['Datetime'] >= start_dt) & (df['Datetime'] < end_dt)
#     return df.loc[mask].copy()

# def summarize_cbm_and_waste(df):
#     """
#     Gruppiert nach 'Dimension' und berechnet:
#       - total_cbm: Summe CBM
#       - waste_cbm: Summe CBM, wo Classification == 'Waste'
#       - waste_percent: Prozentualer Anteil
#     """
#     grouped = df.groupby('Dimension').agg(
#         total_cbm=('CBM', 'sum'),
#         waste_cbm=('CBM', lambda x: x[df.loc[x.index, 'Classification'] == 'Waste'].sum())
#     ).reset_index()
    
#     grouped['waste_percent'] = (grouped['waste_cbm'] / grouped['total_cbm'] * 100).round(2)
#     grouped['total_cbm'] = grouped['total_cbm'].round(3)
#     grouped['waste_cbm'] = grouped['waste_cbm'].round(3)
    
#     return grouped

# def main():
#     filepath = r"C:\Users\jfxbu\OneDrive - Universitaet St.Gallen\Dokumente\St.Gallen\Praktika\Gelo\Produktivitätsbericht\MicroTec\2024_12_5_Formatted.xlsx"
#     df = load_and_prepare_data(filepath)
    
#     # Beispiel: Auftrag A von 06:00 bis 10:00
#     start_dt_A = datetime(2025, 3, 17, 6, 0)
#     end_dt_A   = datetime(2025, 3, 17, 10, 0)
#     df_a = filter_data_for_order(df, start_dt_A, end_dt_A)
#     result_a = summarize_cbm_and_waste(df_a)
#     print("=== Auftrag A (06:00-10:00) ===")
#     print(result_a)
    
#     # Weitere Aufträge können analog hinzugefügt werden

# if __name__ == "__main__":
#     main()


import pandas as pd
from datetime import datetime

def load_and_prepare_data(filepath):
    """
    Lädt das Excel-Dokument mit den vorhandenen Spaltenüberschriften (z.B. "Column1", "Column2", etc.)
    und benennt die relevanten Spalten gemäß deiner Vorgabe:
      - Column2 => 'Year'
      - Column3 => 'Month'
      - Column4 => 'Day'
      - Column5 => 'Hour'
      - Column6 => 'Minute'
      - Column16 => 'Dimension'
      - Column21 => 'Classification'
      - Column27 => 'CBM'
    """
    # Lies das Excel-Dokument ein und verwende die erste Zeile als Header
    df = pd.read_excel(filepath, header=0)
    
    # Debug: Zeige die vorhandenen Spaltennamen
    # print("Vorhandene Spalten:", df.columns.tolist())
    
    # Benenne die relevanten Spalten um
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
    
    # Optional: Falls du noch weitere Spalten hast, kannst du sie hier belassen oder entfernen.
    
    # Erstelle die Datetime-Spalte aus den Zeitkomponenten
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    # print(df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Datetime']].head())


    # Säubere den Dimensions-String (z.B. '17.0 x 100.0 mm' -> '17x100')
    df['Dimension'] = df['Dimension'].apply(normalize_dimension)
    
    return df

def normalize_dimension(dim_str):
    """
    Bereinigt den Dimensions-String.
    Beispiel: '17.0 x 100.0 mm' -> '17x100'
    """
    if pd.isna(dim_str):
        return "Unknown"
    
    # Umwandlung in Kleinbuchstaben, Entfernen von 'mm', Ersetzen von Komma durch Punkt
    dim_str = str(dim_str).lower().replace('mm', '').replace(',', '.')
    # Entferne Leerzeichen und überflüssige Zeichen
    dim_str = dim_str.strip().replace(' ', '').replace('.0', '')
    return dim_str

def filter_data_for_order(df, start_dt, end_dt):
    """
    Filtert das DataFrame auf Zeilen, die in der Zeitspanne [start_dt, end_dt) liegen.
    """
    mask = (df['Datetime'] >= start_dt) & (df['Datetime'] < end_dt)
    return df.loc[mask].copy()

def summarize_cbm_and_waste(df):
    """
    Gruppiert das DataFrame nach 'Dimension' und berechnet:
      - total_cbm: aufsummierte CBM
      - waste_cbm: aufsummierte CBM, bei denen Classification == 'Waste'
      - waste_percent: prozentualer Ausschussanteil (waste_cbm / total_cbm * 100)
    """
    grouped = df.groupby('Dimension').agg(
        total_cbm=('CBM', 'sum'),
        waste_cbm=('CBM', lambda x: x[df.loc[x.index, 'Classification'] == 'Waste'].sum())
    ).reset_index()
    
    grouped['waste_percent'] = (grouped['waste_cbm'] / grouped['total_cbm'] * 100).round(2)
    grouped['total_cbm'] = grouped['total_cbm'].round(3)
    grouped['waste_cbm'] = grouped['waste_cbm'].round(3)
    
    return grouped

def main():
    filepath = r"C:\Users\jfxbu\OneDrive - Universitaet St.Gallen\Dokumente\St.Gallen\Praktika\Gelo\Produktivitätsbericht\MicroTec\2024_12_5_Formatted.xlsx"

    df = load_and_prepare_data(filepath)
    
    #Auftrag 1
    start_dt_A = datetime(2024, 12, 5, 6, 0)
    end_dt_A   = datetime(2024, 12, 5, 11, 25)
    df_a = filter_data_for_order(df, start_dt_A, end_dt_A)
    result_a = summarize_cbm_and_waste(df_a)
    print("=== Auftrag A (06:00-11:25) ===")
    print(result_a)
    
    # Auftrag 2
    start_dt_B = datetime(2024, 12, 5, 11, 24)
    end_dt_B   = datetime(2024, 12, 5, 12, 0)
    df_b = filter_data_for_order(df, start_dt_B, end_dt_B)
    result_b = summarize_cbm_and_waste(df_b)
    print("\n=== Auftrag B (11:20-12:00) ===")
    print(result_b)
    
    # Weitere Aufträge können analog hinzugefügt werden.
    # Optional: Ergebnisse als CSV oder Excel speichern.
    # result_a.to_csv("Auftrag_A_Ergebnis.csv", index=False)
    # result_b.to_csv("Auftrag_B_Ergebnis.csv", index=False)

if __name__ == "__main__":
    main()

