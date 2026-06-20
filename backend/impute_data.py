import pandas as pd
import numpy as np

INPUT_PATH = "jan to may police violation_anonymized791b166.csv"
OUTPUT_PATH = "jan_to_may_police_violation_imputed.csv"

def impute_missing_data():
    print("Step 1: Loading dataset...")
    df = pd.read_csv(INPUT_PATH)
    
    print("\nInitial null values per column:")
    initial_nulls = df.isnull().sum()
    print(initial_nulls[initial_nulls > 0])
    
    # Define 5 grouping features for imputation key
    # We round latitude and longitude to 4 decimals (~11 meters) to align close points
    df['lat_grid'] = df['latitude'].round(4)
    df['lon_grid'] = df['longitude'].round(4)
    
    group_keys = ['lat_grid', 'lon_grid', 'police_station', 'junction_name', 'location']
    
    # Fill any NaNs in the grouping columns first so they can be grouped successfully
    for col in group_keys:
        df[col] = df[col].fillna('UNKNOWN_VAL')
        
    print(f"\nGrouping data by: {group_keys}")
    
    # Columns we want to impute (where missing values are likely to be filled)
    columns_to_impute = [
        'vehicle_type', 'violation_type', 'offence_code', 
        'validation_status', 'device_id', 'created_by_id', 'center_code'
    ]
    
    # Perform grouping and fillna using the group mode
    print("Applying group-by mode imputation. This might take a moment...")
    
    # Fast grouped imputation: 
    # For each group, we compute the mode. If a value is null, we fill it with the group mode.
    # To run this efficiently on 300k rows:
    for col in columns_to_impute:
        if col in df.columns:
            print(f"Imputing {col}...")
            # Fast vectorized MultiIndex mapping:
            # 1. Create a temp dataframe with the group keys as index
            df_temp = df[group_keys + [col]].copy()
            df_temp = df_temp.set_index(group_keys)
            
            # 2. Get the first non-null value per group
            group_map = df_temp[col].dropna().groupby(level=group_keys).first()
            
            # 3. Fill nulls using pandas index alignment
            df_temp[col] = df_temp[col].fillna(group_map)
            
            # 4. Write back to original dataframe
            df[col] = df_temp[col].values

    # Revert 'UNKNOWN_VAL' place holders back to NaN if they weren't originally there
    for col in group_keys:
        df[col] = df[col].replace('UNKNOWN_VAL', np.nan)
        
    # Drop helper columns
    df = df.drop(columns=['lat_grid', 'lon_grid'])

    print("\nNull values after imputation:")
    final_nulls = df.isnull().sum()
    print(final_nulls[final_nulls > 0])
    
    # Compare before and after
    print("\nImputation summary:")
    for col in columns_to_impute:
        if col in df.columns:
            diff = initial_nulls[col] - final_nulls[col]
            print(f" - {col}: Filled {diff} null values ({initial_nulls[col]} -> {final_nulls[col]})")

    print(f"\nStep 3: Saving imputed dataset to '{OUTPUT_PATH}'...")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Imputation pipeline completed successfully!")

if __name__ == "__main__":
    impute_missing_data()
