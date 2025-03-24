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


# --- Beispielnutzung ---
#12.03
#file_path = r"C:\Users\jfxbu\OneDrive - Universitaet St.Gallen\Dokumente\St.Gallen\Praktika\Gelo\Produktivitätsbericht\HewSaw\Automation\12.03\13.03.2025_Produktivitätsbericht.pdf"
#05.12
#file_path = r"C:\Users\jfxbu\OneDrive - Universitaet St.Gallen\Dokumente\St.Gallen\Praktika\Gelo\Produktivitätsbericht\HewSaw\Produkivitätsberichte\051224_Produktivitätsbericht.pdf"
#04.03
#file_path = r"C:\Users\jfxbu\OneDrive - Universitaet St.Gallen\Dokumente\St.Gallen\Praktika\Gelo\Produktivitätsbericht\HewSaw\Automation\04.03.25\040325_Produktivitätsbericht.pdf"

#06.03
file_path = r"C:\Users\jfxbu\OneDrive - Universitaet St.Gallen\Dokumente\Coding\Gelo\Data\06.03.25\060325_Produktivitätsbericht.pdf"

df_parsed = extract_table_with_suborders_clean(file_path)


# Umwandlung der Spalten in Zahlen (wie gehabt)
numeric_columns = ['stämme', 'vol_eingang', 'durchschn_stammlänge', 'teile', 'vol_ausgang']

for col in numeric_columns:
    df_parsed[col] = (
        df_parsed[col]
        .replace('', pd.NA)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    df_parsed[col] = pd.to_numeric(df_parsed[col], errors='coerce')

# Apply transformations
df_parsed["vol_eingang"] = df_parsed["vol_eingang"] / 1000
df_parsed["durchschn_stammlänge"] = df_parsed["durchschn_stammlänge"] / 100
df_parsed["vol_ausgang"] = df_parsed["vol_ausgang"] / 100

print(df_parsed)

# In Excel speichern
output_dir = os.path.dirname(file_path)
output_file_path = os.path.join(output_dir, "Parsed_Tabelle_final_Dientstag.xlsx")
df_parsed.to_excel(output_file_path, index=False)
print(f"Datei '{output_file_path}' wurde gespeichert.")