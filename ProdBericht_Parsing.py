import pdfplumber
import pandas as pd
import re
import os

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
    
    # Regex für Haupt- und Unterzeilen
    main_row_pattern = re.compile(
        r"(?P<auftrag>\d{5}\s*-\s*.*?)(?=\s+\d)"  # Erfasst ab dem ersten Auftrags-Code (5 Ziffern, Bindestrich)
        r"\s+(?P<stämme>[\d.,]+)\s+"
        r"(?P<vol_eingang>[\d.,]+)\s+"
        r"(?P<durchschn_stammlänge>[\d.,]+)\s+"
        r"(?P<teile>[\d.,]+)\s+"
        r"(?P<vol_ausgang>[\d.,]+)"
        r"(?:\s+.*)?"
    )
    
    sub_row_pattern = re.compile(
        r"(?P<unterauftrags_muster>\d+x\d+)\s+"
        r"(?P<teile>[\d.,]+)\s+"
        r"(?P<vol_ausgang>[\d.,]+)"
    )
    
    # Verbesserter Merge-Algorithmus:
    merged_lines = []
    buffer = ""
    for line in lines:
        # Falls Zeile als Unterzeile erkannt wird, Puffer (falls vorhanden) abschließen und Unterzeile speichern
        if sub_row_pattern.match(line):
            if buffer:
                # Vor dem Speichern den Puffer auf bereinigte Hauptzeile prüfen:
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
                # Puffer vor dem Speichern bereinigen, falls er Headertext enthält
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
            current_main_order = main_match.groupdict()
            row = {
                'auftrag': current_main_order['auftrag'].strip(),
                'unterkategorie': "",
                'stämme': current_main_order['stämme'],
                'vol_eingang': current_main_order['vol_eingang'],
                'durchschn_stammlänge': current_main_order['durchschn_stammlänge'],
                'teile': current_main_order['teile'],
                'vol_ausgang': current_main_order['vol_ausgang'],
            }
            result_rows.append(row)
        else:
            # Versuch als Unterzeile zu parsen, falls es aktuell einen Hauptauftrag gibt
            if current_main_order:
                sub_match = sub_row_pattern.match(line)
                if sub_match:
                    sub_row = sub_match.groupdict()
                    full_row = {
                        'auftrag': current_main_order['auftrag'].strip(),
                        'unterkategorie': sub_row['unterauftrags_muster'],
                        'stämme': "",
                        'vol_eingang': "",
                        'durchschn_stammlänge': "",
                        'teile': sub_row['teile'],
                        'vol_ausgang': sub_match.group('vol_ausgang'),
                    }
                    result_rows.append(full_row)
    return pd.DataFrame(result_rows)


# --- Beispielnutzung ---
#12.03
#file_path = r"C:\Users\jfxbu\OneDrive - Universitaet St.Gallen\Dokumente\St.Gallen\Praktika\Gelo\Produktivitätsbericht\HewSaw\Automation\12.03\13.03.2025_Produktivitätsbericht.pdf"

#05.12
#file_path = r"C:\Users\jfxbu\OneDrive - Universitaet St.Gallen\Dokumente\St.Gallen\Praktika\Gelo\Produktivitätsbericht\HewSaw\Produkivitätsberichte\051224_Produktivitätsbericht.pdf"

#04.03
file_path = r"C:\Users\jfxbu\OneDrive - Universitaet St.Gallen\Dokumente\St.Gallen\Praktika\Gelo\Produktivitätsbericht\HewSaw\Automation\04.03.25\040325_Produktivitätsbericht.pdf"

df_parsed = extract_table_with_suborders_clean(file_path)


# Liste der Spalten, die in Zahlen umgewandelt werden sollen (ab Spalte B)
numeric_columns = ['stämme', 'vol_eingang', 'durchschn_stammlänge', 'teile', 'vol_ausgang']

# Umwandeln: Komma -> Punkt und dann float
# Umwandeln: Komma -> Punkt und dann float (ab Spalte B)
for col in numeric_columns:
    df_parsed[col] = (
        df_parsed[col]
        .replace('', pd.NA)  # Leere Strings in NaN umwandeln
        .str.replace('.', '', regex=False)  # Tausenderpunkt entfernen
        .str.replace(',', '.', regex=False)  # Komma als Dezimalpunkt
    )
    df_parsed[col] = pd.to_numeric(df_parsed[col], errors='coerce')

# Apply transformations
df_parsed["vol_eingang"] = df_parsed["vol_eingang"] / 1000
df_parsed["durchschn_stammlänge"] = df_parsed["durchschn_stammlänge"] / 100
df_parsed["vol_ausgang"] = df_parsed["vol_ausgang"] / 100

print(df_parsed)


# In Excel speichern
output_dir = os.path.dirname(file_path)
output_file_path = os.path.join(output_dir, "Parsed_Tabelle_final.xlsx")
df_parsed.to_excel(output_file_path, index=False)
print(f"Datei '{output_file_path}' wurde gespeichert.")