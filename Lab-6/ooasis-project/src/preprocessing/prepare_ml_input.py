import pandas as pd
import numpy as np
import os

def main():
    file_path = os.path.join("data", "processed", "observations_valid.csv")
    
    print("=== ML Input Preparation: Loading and Conversion ===")
    print(f"Input file: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        records_loaded = len(df)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Please ensure the file exists.")
        return

    required_cols = ['temperature', 'velocity', 'altitude', 'signal_strength']
    
    # ---------------------------------------------------------
    # TASK 1: CLEANING & CONVERSION
    # ---------------------------------------------------------
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    available_required_cols = [c for c in required_cols if c in df.columns]
    df_clean = df.dropna(subset=available_required_cols).copy()
    
    if 'altitude' in df_clean.columns:
        df_clean = df_clean[df_clean['altitude'] >= 0]
        
    records_accepted = len(df_clean)
    records_rejected = records_loaded - records_accepted
    
    print(f"Records loaded: {records_loaded}")
    print(f"Records accepted: {records_accepted}")
    print(f"Records rejected: {records_rejected}")

    # ---------------------------------------------------------
    # TASK 2: NORMALIZATION
    # ---------------------------------------------------------
    print("\n=== ML Input Preparation: Normalization ===")
    
    for col in available_required_cols:
        col_min = df_clean[col].min()
        col_max = df_clean[col].max()
        
        if col_max - col_min == 0:
            df_clean[col] = 0.0
        else:
            df_clean[col] = (df_clean[col] - col_min) / (col_max - col_min)
            
    print("Normalization completed successfully.")
    print("All selected numerical features are in range [0,1].")

    # ---------------------------------------------------------
    # TASK 3: DERIVED FEATURES
    # ---------------------------------------------------------
    print("\n=== ML Input Preparation: Derived Features ===")
    
    if 'temperature' in df_clean.columns and 'velocity' in df_clean.columns:
        df_clean['temperature_velocity_interaction'] = df_clean['temperature'] * df_clean['velocity']
        
    if 'altitude' in df_clean.columns and 'signal_strength' in df_clean.columns:
        df_clean['altitude_signal_ratio'] = np.where(
            df_clean['signal_strength'] == 0,
            0.0,  
            df_clean['altitude'] / df_clean['signal_strength']
        )
        
    print("New features added:")
    print("- temperature_velocity_interaction")
    print("- altitude_signal_ratio")

    # ---------------------------------------------------------
    # TASK 4: TEMPORAL FEATURES
    # ---------------------------------------------------------
    print("\n=== ML Input Preparation: Temporal Features ===")
    
    if 'timestamp' in df_clean.columns:
        temp_datetime = pd.to_datetime(df_clean['timestamp'], errors='coerce')
        df_clean['hour_normalized'] = temp_datetime.dt.hour / 24.0
        
    print("New feature added:")
    print("- hour_normalized")
    
    # ---------------------------------------------------------
    # TASK 5: FEATURE SELECTION
    # ---------------------------------------------------------
    print("\n=== ML Input Preparation: Feature Selection ===")
    
    final_features_list = [
        'temperature',
        'velocity',
        'altitude',
        'signal_strength',
        'temperature_velocity_interaction',
        'altitude_signal_ratio',
        'hour_normalized'
    ]
    
    available_final_features = [f for f in final_features_list if f in df_clean.columns]
    df_final = df_clean[available_final_features].copy()
    
    print("Selected features:")
    for feature in available_final_features:
        print(f"- {feature}")

    # ---------------------------------------------------------
    # TASK 6: TARGET LABELS AND SAVING
    # ---------------------------------------------------------
    print("\n=== ML Input Preparation: Saving Outputs ===")
    
    # Define paths
    features_path = os.path.join("data", "processed", "model_features.csv")
    labels_path = os.path.join("data", "processed", "model_labels.csv")
    
    # Extract labels from the cleaned dataframe (preserves exact record order)
    if 'anomaly_flag' in df_clean.columns:
        df_labels = df_clean[['anomaly_flag']].copy()
    else:
        print("Error: 'anomaly_flag' not found in dataset.")
        return

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    
    # Save files without the pandas row indices
    df_final.to_csv(features_path, index=False)
    df_labels.to_csv(labels_path, index=False)
    
    # Print final validation metrics
    print(f"Saved file: {features_path}")
    print(f"Saved file: {labels_path}")
    print(f"\nNumber of records: {len(df_final)}")
    print(f"Number of features: {len(df_final.columns)}")
    
    print("\nExample label record:")
    if not df_labels.empty:
        # Prints out native Python dictionary format matching expectations
        print(df_labels.iloc[0].to_dict())

if __name__ == "__main__":
    main()