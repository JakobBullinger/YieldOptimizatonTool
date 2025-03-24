import pdfplumber
import pandas as pd
import re
import os

# Pfad zur PDF
#12.03
file_path = r"C:\Users\jfxbu\OneDrive - Universitaet St.Gallen\Dokumente\St.Gallen\Praktika\Gelo\Produktivitätsbericht\HewSaw\Automation\12.03\13.03.2025_Produktivitätsbericht.pdf"

#05.12
# file_path = r"C:\Users\jfxbu\OneDrive - Universitaet St.Gallen\Dokumente\St.Gallen\Praktika\Gelo\Produktivitätsbericht\HewSaw\Produkivitätsberichte\051224_Produktivitätsbericht.pdf"

#04.03
#file_path = r"C:\Users\jfxbu\OneDrive - Universitaet St.Gallen\Dokumente\St.Gallen\Praktika\Gelo\Produktivitätsbericht\HewSaw\Automation\04.03.25\040325_Produktivitätsbericht.pdf"


def extract_table_with_suborders_clean(file_path, start_keyword="Auftrag"):
    with pdfplumber.open(file_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"

    # Suche nach Startpunkt "Auftrag"
    start_index = full_text.find(start_keyword)
    if start_index == -1:
        raise ValueError("Startkeyword 'Auftrag' nicht gefunden!")

    # Tabelle ab Auftrag
    table_text = full_text[start_index:]
    
    for line in table_text.splitlines():
        print(repr(line))

    # Auftrags-Header und Sub-Zeilen trennen
    # Neuer Hauptauftrag Regex (erkennt auch "mit Nut" etc.)
    main_row_pattern = re.compile(
        r"(?P<auftrag>\d{5}\s?-\s?.*?)\s+"  # Auftrag
        r"(?P<stämme>[\d.,]+)\s+"
        r"(?P<vol_eingang>[\d.,]+)\s+"
        r"(?P<durchschn_stammlänge>[\d.,]+)\s+"
        r"(?P<teile>[\d.,]+)\s+"
        r"(?P<vol_ausgang>[\d.,]+)"
    )


    # Unterzeilen Muster (z.B. 17x95  10292  51.88)
    sub_row_pattern = re.compile(
        r"(?P<unterauftrags_muster>\d+x\d+)\s+"
        r"(?P<teile>[\d.,]+)\s+"
        r"(?P<vol_ausgang>[\d.,]+)"
    )
    

    result_rows = []
    current_main_order = None  # Aktueller Auftrag für Unterzeilen

    # Zeilen einzeln durchgehen
    for line in table_text.splitlines():
        line = line.strip()
        if not line:
            continue  # Leere Zeilen überspringen

        # Prüfen ob Hauptauftrag
        main_match = main_row_pattern.match(line)
        if main_match:
            current_main_order = main_match.groupdict()
            # Jetzt aber "unterkategorie" leer für Hauptauftrag
            row = {
                'auftrag': current_main_order['auftrag'],
                'unterkategorie': "",  # leer, weil Hauptauftrag
                'stämme': current_main_order['stämme'],
                'vol_eingang': current_main_order['vol_eingang'],
                'durchschn_stammlänge': current_main_order['durchschn_stammlänge'],
                'teile': current_main_order['teile'],
                'vol_ausgang': current_main_order['vol_ausgang'],
            }
            result_rows.append(row)
        else:
            # Prüfen ob Unterauftrag zur aktuellen Bestellung
            if current_main_order:
                sub_match = sub_row_pattern.match(line)
                if sub_match:
                    sub_row = sub_match.groupdict()
                    # Leere Werte für die ersten Spalten, um es übersichtlicher zu machen
                    full_row = {
                        'auftrag': current_main_order['auftrag'],
                        'unterkategorie': sub_row['unterauftrags_muster'],  # hier kommt die Unterkategorie rein
                        'stämme': "",  # leer lassen
                        'vol_eingang': "",
                        'durchschn_stammlänge': "",
                        'teile': sub_row['teile'],
                        'vol_ausgang': sub_row['vol_ausgang'],
                        'durchschn_länge': "",
                        'ausbeute': "",
                        'prod_ratio': ""
                    }
                    result_rows.append(full_row)

    # DataFrame erstellen
    df = pd.DataFrame(result_rows)

    return df



# Tabelle extrahieren
df_parsed = extract_table_with_suborders_clean(file_path)

# Liste der Spalten, die entfernt werden sollen
columns_to_drop = ['durchschn_länge','ausbeute', 'prod_ratio']



# DataFrame ohne diese Spalten
df_final = df_parsed.drop(columns=columns_to_drop)

# Liste der Spalten, die in Zahlen umgewandelt werden sollen (ab Spalte B)
numeric_columns = ['stämme', 'vol_eingang', 'durchschn_stammlänge', 'teile', 'vol_ausgang']

# Umwandeln: Komma -> Punkt und dann float
# Umwandeln: Komma -> Punkt und dann float (ab Spalte B)
for col in numeric_columns:
    df_final[col] = (
        df_final[col]
        .replace('', pd.NA)  # Leere Strings in NaN umwandeln
        .str.replace('.', '', regex=False)  # Tausenderpunkt entfernen
        .str.replace(',', '.', regex=False)  # Komma als Dezimalpunkt
    )
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

# Apply transformations
df_final["vol_eingang"] = df_final["vol_eingang"] / 1000
df_final["durchschn_stammlänge"] = df_final["durchschn_stammlänge"] / 100
df_final["vol_ausgang"] = df_final["vol_ausgang"] / 100

# Optional: Anzeige zur Kontrolle
print(df_final.head())

# In Excel speichern
output_dir = os.path.dirname(file_path)
output_file_path = os.path.join(output_dir, "Parsed_Tabelle_ohne_Letzte_Spalten_02.xlsx")
df_final.to_excel(output_file_path, index=False)
print(f"Datei '{output_file_path}' wurde gespeichert.")
