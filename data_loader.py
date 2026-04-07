import pandas as pd
import numpy as np
import datetime as dt

# Code -> label mapped globally to use in the app
classification_map = {1:'Fatal', 2:'Grievous', 3:'Minor', 4:'Non-Injury/Property'}
causes_map = {1:'Over-Speeding', 2:'Driver Negligence', 3:'Road Defect', 4:'Weather', 5:'Other'}
road_feature_map = {0:'Straight', 2:'Curve/Bend', 3:'Bridge/Culvert', 4:'Junction', 5:'Other'}
road_cond_map = {1:'Dry', 2:'Wet', 3:'Under Repair', 4:'Slippery', 5:'Other'}
weather_map = {1:'Clear', 3:'Rainy', 4:'Foggy', 5:'Other'}
nature_map = {1:'Head-on Collision', 2:'Rear-end Collision', 3:'Side Impact', 4:'Hit Pedestrian',
              5:'Rollover', 6:'Hit Animal', 7:'Hit Object', 8:'Other'}

def safe_map(series, mapping):
    return pd.to_numeric(series, errors='coerce').round(0).astype('Int64').map(
        lambda x: mapping.get(x, 'Unknown') if pd.notna(x) else 'Unknown')

def parse_hour(t):
    if pd.isna(t): return np.nan
    t = str(t).strip()
    for fmt in ('%H:%M:%S', '%H:%M', '%I:%M %p'):
        try:
            return dt.datetime.strptime(t, fmt).hour
        except: pass
    try: return int(float(t))
    except: return np.nan

def load_and_clean_data(file_path):
    COL_MAP = {
        0: 'SNo', 1: 'Date', 2: 'Time', 3: 'Location_NH53', 4: 'Location_NH6',
        5: 'Accident_Location', 6: 'Nature_Code', 7: 'Classification_Code',
        8: 'Causes_Code', 9: 'Road_Feature_Code', 10: 'Road_Condition_Code',
        11: 'Intersection_Type_Code', 12: 'Weather_Code',
        13: 'LightVehicle', 14: 'HeavyVehicle', 15: 'Bus',
        16: 'Motorcycle', 17: 'Cycle', 18: 'Pedestrian',
        19: 'Vehicle_RegNo', 20: 'Vehicle_Responsible',
        21: 'Fatal_Count', 22: 'Grievous_Count', 23: 'Minor_Count',
        24: 'NonInjured_Count', 25: 'Animals_Killed',
        26: 'Help_Given', 27: 'Remarks',
        28: 'Latitude', 29: 'Longitude', 30: 'Chainage'
    }

    xl = pd.ExcelFile(file_path)
    dfs = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=None, skiprows=4)
        df.columns = range(len(df.columns))
        df = df.dropna(how='all')
        df = df[pd.to_numeric(df[0], errors='coerce').notna()].copy()
        df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})
        df['Year'] = int(sheet)
        dfs.append(df)

    master = pd.concat(dfs, ignore_index=True)
    
    # Cleaning
    master['Date'] = pd.to_datetime(master['Date'], errors='coerce')
    master['Month'] = master['Date'].dt.month
    master['DayOfWeek'] = master['Date'].dt.dayofweek
    master['Hour'] = master['Time'].apply(parse_hour)

    master['Classification'] = safe_map(master['Classification_Code'], classification_map)
    master['Causes'] = safe_map(master['Causes_Code'], causes_map)
    master['Road_Feature'] = safe_map(master['Road_Feature_Code'], road_feature_map)
    master['Road_Condition'] = safe_map(master['Road_Condition_Code'], road_cond_map)
    master['Weather'] = safe_map(master['Weather_Code'], weather_map)
    master['Nature'] = safe_map(master['Nature_Code'], nature_map)

    for col in ['Fatal_Count', 'Grievous_Count', 'Minor_Count', 'NonInjured_Count', 'Animals_Killed',
                'LightVehicle', 'HeavyVehicle', 'Bus', 'Motorcycle', 'Cycle', 'Pedestrian']:
        master[col] = pd.to_numeric(master[col], errors='coerce').fillna(0).astype(int)

    master['Severity_Score'] = master['Fatal_Count']*5 + master['Grievous_Count']*3 + master['Minor_Count']*1
    master['Latitude'] = pd.to_numeric(master['Latitude'], errors='coerce')
    master['Longitude'] = pd.to_numeric(master['Longitude'], errors='coerce')

    master['Accident_Location'] = (master['Accident_Location']
        .astype(str).str.strip().str.title()
        .replace({'Rural': 'Rural Area', 'Urban': 'Urban Area', 'Nan': 'Unknown'}))

    # Keep only valid coordinates for maps/analysis
    master_clean = master.dropna(subset=['Latitude', 'Longitude', 'Date']).copy()
    master_clean = master_clean[(master_clean['Latitude'] > 20) & (master_clean['Latitude'] < 22)]
    master_clean = master_clean[(master_clean['Longitude'] > 72) & (master_clean['Longitude'] < 74)]

    return master_clean
