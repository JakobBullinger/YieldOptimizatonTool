import pandas as pd
from datetime import datetime

# 1) Define your mapping dictionary.
synonyms = {
    "17x95": "17x100",
    "23x100": "23x103",
    "47x212": "47x221"
}

def unify_dimension(dim_str):
    """
    Looks up dim_str in the synonyms dictionary. 
    If found, returns the mapped dimension. Otherwise returns dim_str unchanged.
    """
    return synonyms.get(dim_str, dim_str)

def normalize_dimension(dim_str):
    """
    Converts strings like '17.0 x 100.0 mm' to a uniform format, e.g. '17x100'.
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

def load_and_prepare_data(filepath):
    """
    Reads your CSV file (without header) and prepares the data:
      - Reads the file with header=None.
      - Renames columns based on their index:
          Index 1  -> 'Year'
          Index 2  -> 'Month'
          Index 3  -> 'Day'
          Index 4  -> 'Hour'
          Index 5  -> 'Minute'
          Index 15 -> 'Dimension'
          Index 21 -> 'Classification'
          Index 26 -> 'CBM'
      - Creates a Datetime column
      - Normalizes and unifies dimension names
    """
    # Read CSV without header
    df = pd.read_csv(filepath, header=None, sep=';')
    
    # Rename columns based on index positions:
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
    
    # Create the Datetime column
    df['Datetime'] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute']])
    
    # Normalize dimension strings and apply mapping
    df['Dimension'] = df['Dimension'].apply(normalize_dimension).apply(unify_dimension)
    
    return df

def filter_data_for_order(df, start_dt, end_dt, dimensions):
    """
    Filters the DataFrame by:
      1) Time: [start_dt, end_dt)
      2) Dimension: only the given dimensions.
    """
    mask = (
        (df['Datetime'] >= start_dt) &
        (df['Datetime'] < end_dt) &
        (df['Dimension'].isin(dimensions))
    )
    return df.loc[mask].copy()

def summarize_cbm_and_waste(df):
    """
    Groups the DataFrame by 'Dimension' and calculates:
      - total_cbm: sum of CBM
      - waste_cbm: sum of CBM where Classification == 'Waste'
      - waste_percent: percentage of waste (waste_cbm / total_cbm * 100)
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

def main():
    """
    Main function:
      1) Load and prepare data from CSV
      2) Define order info (time windows + dimensions)
      3) Filter and summarize for each order
      4) Print results
    """
    # Passe den Pfad zu deiner CSV-Datei an
    
    filepath = r"C:\Users\jfxbu\OneDrive - Universitaet St.Gallen\Dokumente\Coding\Gelo\Data\05.12.24\2024_12_5.csv"
    df = load_and_prepare_data(filepath)
    
    orders = {
        "Auftrag A": {
            "time_window": (datetime(2024, 12, 5, 6, 0),
                            datetime(2024, 12, 5, 11, 25)),
            "dimensions": ["17x100", "47x135"]
        },
        "Auftrag B": {
            "time_window": (datetime(2024, 12, 5, 11, 23),
                            datetime(2024, 12, 5, 12, 0)),
            "dimensions": ["76x96"]
        }
    }
    
    for order_name, order_info in orders.items():
        start_dt, end_dt = order_info["time_window"]
        dims = order_info["dimensions"]
        
        df_order = filter_data_for_order(df, start_dt, end_dt, dims)
        result = summarize_cbm_and_waste(df_order)
        
        print(f"=== {order_name} ===")
        print(result)
        print()

if __name__ == "__main__":
    main()
