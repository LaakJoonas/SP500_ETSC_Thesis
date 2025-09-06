
#S&P500 ETSC with 6 datasets

import pandas as pd
import numpy as np
import os

#Data loading functions
def load_sp500_data(filepath):
    print("Loading SP500 data...")
    
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    
    if any(word in first_line.lower() for word in ['date', 'time', 'open', 'high', 'low', 'close', 'volume']):
        df = pd.read_csv(filepath, names=['id', 'datetime_str', 'open', 'high', 'low', 'close', 'volume', 'count', 'vwap'], skiprows=1)
    else:
        df = pd.read_csv(filepath, names=['id', 'datetime_str', 'open', 'high', 'low', 'close', 'volume', 'count', 'vwap'])
    
    # parse timestamps
    df['timestamp'] = pd.to_datetime(df['datetime_str'], errors='coerce')
    
    # Convert to numeric
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'count', 'vwap']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean and sort
    df = df.dropna().sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
    
    print(f"Loaded {len(df):,} SP500 records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap']]

def load_vix_data(filepath):
    #Load only open values
    print("Loading VIX data...")
    try:
        vix_df = pd.read_csv(filepath)
        vix_df['DATE'] = pd.to_datetime(vix_df['DATE'], format='%m/%d/%Y')
        vix_df = vix_df[['DATE', 'OPEN']].rename(columns={'OPEN': 'vix_open'}).dropna()
        
        print(f"Loaded {len(vix_df):,} VIX records from {vix_df['DATE'].min()} to {vix_df['DATE'].max()}")
        return vix_df
    except Exception as e:
        print(f"Error loading VIX data: {e}")
        return pd.DataFrame()

def load_macro_data(filepath):
    # macroeconomic indicators
    print("Loading macro indicators...")
    
    try:
        if filepath.endswith(('.xls', '.xlsx')):
            macro_df = pd.read_excel(filepath)
        else:
            macro_df = pd.read_csv(filepath)
        
        macro_df['DATE'] = pd.to_datetime(macro_df['DATE'])
        
        # Select the 4 macro indicators
        key_indicators = ['unrate', 'ffer', 'indpro', 'ccpi']
        available_indicators = [col for col in key_indicators if col in macro_df.columns]
        
        if not available_indicators:
            print("No valid macro indicators found")
            return pd.DataFrame()
        
        macro_df = macro_df[['DATE'] + available_indicators].dropna()
        
        print(f"Loaded {len(macro_df):,} macro records with indicators: {available_indicators}")
        return macro_df
    except Exception as e:
        print(f"Error loading macro data: {e}")
        return pd.DataFrame()

def load_technical_indicators(filepath):
    #Load the 4 technical indicators: EMA-26, RSI-14, MACD histogram, CCI-20
    print("Loading technical indicators...")
    
    try:
        # Read in chunks from the very large file, otherwise crash
        chunk_size = 100000
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            chunks.append(chunk)
        
        indicators_df = pd.concat(chunks, ignore_index=True)
        indicators_df['timestamp'] = pd.to_datetime(indicators_df['timestamp'])
        
        # Select the 4 indicators
        required_indicators = ['ema_26', 'rsi_14', 'macd_histogram', 'cci']
        available_indicators = [col for col in required_indicators if col in indicators_df.columns]
        
        if not available_indicators:
            print("No valid technical indicators found!")
            return pd.DataFrame()
        
        indicators_df = indicators_df[['timestamp'] + available_indicators].dropna(how='all')
        
        print(f"Loaded {len(indicators_df):,} indicator records: {available_indicators}")
        return indicators_df
    except Exception as e:
        print(f"Error loading technical indicators: {e}")
        return pd.DataFrame()

# Data merge functions

def merge_vix_data(price_df, vix_df):
    # Daily VIX data with minute-level price data
    if vix_df.empty:
        return price_df
    
    print("Merging VIX data...")
    
    # Create date columns for merging
    price_df['date'] = price_df['timestamp'].dt.date
    vix_df['date'] = vix_df['DATE'].dt.date
    
    # Merge and forward fill
    merged_df = price_df.merge(vix_df[['date', 'vix_open']], on='date', how='left')
    merged_df['vix_open'] = merged_df['vix_open'].ffill()
    
    print(f"VIX merge: {merged_df['vix_open'].notna().sum():,} rows have VIX data")
    return merged_df.drop('date', axis=1)

def merge_macro_data(price_df, macro_df, lag_months=2):
    """Merge monthly macro data with minute-level price data (with publication lag)"""
    if macro_df.empty:
        return price_df
    
    print(f"Merging macro data with {lag_months}-month lag...")
    
    # Apply publication lag
    macro_lagged = macro_df.copy()
    macro_lagged['DATE'] = macro_lagged['DATE'] + pd.DateOffset(months=lag_months)
    
    # Create year-month columns for merging
    price_df['year_month'] = price_df['timestamp'].dt.to_period('M')
    macro_lagged['year_month'] = macro_lagged['DATE'].dt.to_period('M')
    
    # Get macro columns
    macro_cols = [col for col in macro_lagged.columns if col not in ['DATE', 'year_month']]
    
    # Merge and keep only complete data
    merged_df = price_df.merge(macro_lagged[['year_month'] + macro_cols], on='year_month', how='left')
    merged_df = merged_df.dropna(subset=macro_cols)
    
    print(f"Macro merge: {len(merged_df):,} rows have complete macro data")
    return merged_df.drop('year_month', axis=1)

def merge_indicator_data(price_df, indicators_df):
    #Merge minute-level technical indicators too price data
    if indicators_df.empty:
        return price_df
    
    print("Merging technical indicators...")
    
    #Timestamp merge
    merged_df = price_df.merge(indicators_df, on='timestamp', how='left')
    
    # Forward fill missing indicator values
    indicator_cols = [col for col in indicators_df.columns if col != 'timestamp']
    for col in indicator_cols:
        merged_df[col] = merged_df[col].ffill()
    
    print(f"Indicators merge: {len(merged_df):,} rows processed")
    return merged_df

# Sequence processing

def uniform_sequences(sequences, max_length=390):
    if not sequences:
        return np.array([])
    
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padding = np.tile(seq[-1], (max_length - len(seq), 1))
            padded_seq = np.vstack([seq, padding])
        else:
            padded_seq = seq[:max_length]
        padded_sequences.append(padded_seq)
    
    return np.array(padded_sequences)

#This generates the .ts file, which is used by the https://timeseriesclassification.com archive and CALIMERA pre made algorithm.
def write_ts_file(filename, X, y, feature_names, problem_name):
    # Write TS format file
    n_samples, n_dimensions, series_length = X.shape
    unique_labels = sorted(set(y))
    
    with open(filename, 'w') as f:
        f.write(f"@problemName {problem_name}\n")
        f.write("@timeStamps false\n")
        f.write("@missing false\n")
        f.write("@univariate false\n")
        f.write(f"@dimensions {n_dimensions}\n")
        f.write("@equalLength true\n")
        f.write(f"@seriesLength {series_length}\n")
        f.write(f"@classLabel true {' '.join(map(str, unique_labels))}\n")
        f.write("@data\n")
        
        for i in range(n_samples):
            for dim in range(n_dimensions):
                f.write(','.join(map(str, X[i, dim, :])))
                if dim < n_dimensions - 1:
                    f.write(':')
            f.write(f":{y[i]}\n")

#Create daily sequences.for 2-class problem when price data is included (!TECH_ONLY)

def create_daily_sequences(df, max_sequence_length=390):
    df['date'] = df['timestamp'].dt.date
    daily_sequences = []
    labels = []
    sequence_dates = []
    print(f"Processing {len(df['date'].unique())} unique trading days...")

    price_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap']
    excluded_cols = ['timestamp', 'date']
    
    all_feature_cols = [col for col in df.columns if col not in excluded_cols]
    
    print(f"Features: {all_feature_cols}")
    print(f"Total features: {len(all_feature_cols)}")
    
    for date, group in df.groupby('date'):
        if len(group) < 50:  # Skip incomplete days with way too little information
            continue
            
        group = group.sort_values('timestamp')
        
        # Alignment to 390 minutes (7:30 to 13:59)
        try:
            # 390-minute time range
            start_datetime = pd.to_datetime(f"{date} 07:30:00")
            end_datetime = pd.to_datetime(f"{date} 13:59:00")
            full_range = pd.date_range(start=start_datetime, end=end_datetime, freq='1min')
            
            # Remove duplicates, and index
            group_clean = group.drop_duplicates(subset=['timestamp'], keep='last')
            group_indexed = group_clean.set_index('timestamp')
            
            # Reindex to full trading day
            aligned_data = group_indexed.reindex(full_range)
            
            # bfill/ffill missing values for all columns
            aligned_data = aligned_data.ffill().bfill()
            
            # Reset index
            aligned_data = aligned_data.reset_index()
            aligned_data = aligned_data.rename(columns={'index': 'timestamp'})
            
            group = aligned_data
            
        except Exception as e:
            print(f"Error aligning {date}: {e}")
            continue
        
        # Truncate to max_length
        if max_sequence_length and len(group) > max_sequence_length:
            group = group.iloc[:max_sequence_length]
        
        # Check for missing information
        missing_features = [col for col in all_feature_cols if col not in group.columns]
        if missing_features:
            print(f"Warning: Missing features for {date}: {missing_features}")
            continue
            
        sequence = group[all_feature_cols].values
        
        # Handle NaN vals
        if np.isnan(sequence).any():
            nan_count = np.isnan(sequence).sum()
            total_values = sequence.size
            if nan_count / total_values > 0.1:  # Skip if >10% NaN as data is bad
                print(f"Skipping {date}: too many NaN values ({nan_count}/{total_values})")
                continue
            # f/bfill NaN values to fix missing day data
            sequence = pd.DataFrame(sequence, columns=all_feature_cols).ffill().bfill().values
        
        # Calculate daily return (up/down) for labeling
        first_close = group['close'].iloc[0]
        last_close = group['close'].iloc[-1]
        daily_return = (last_close - first_close) / first_close
        
        # Binary classification: "Up" (1.0) vs "Down" (2.0), 0-days marked as loss
        label = 1.0 if daily_return > 0 else 2.0
        
        daily_sequences.append(sequence)
        labels.append(label)
        sequence_dates.append(date)
    
    print(f"Created {len(daily_sequences)} sequences")
    
    # Show class distribution for information
    up_days = labels.count(1.0)
    down_days = labels.count(2.0)
    total_days = len(labels)
    print(f"UP days: {up_days} ({up_days/total_days*100:.1f}%)")
    print(f"DOWN days: {down_days} ({down_days/total_days*100:.1f}%)")
    
    return daily_sequences, labels, sequence_dates

#This is almost the same as the previous but it was easier to create a new one rather than old.
def create_daily_sequences_tech_only(df, feature_cols, max_sequence_length=390):
    #Create daily sequences using ONLY TECH (excludes price data from features)
    df['date'] = df['timestamp'].dt.date
    daily_sequences = []
    labels = []
    sequence_dates = []
    
    print(f"Processing {len(df['date'].unique())} unique trading days...")
    print(f"TECH_ONLY Features: {feature_cols}")
    print(f"Total features: {len(feature_cols)}")
    
    for date, group in df.groupby('date'):
        if len(group) < 50:  # Skip bad days
            continue
            
        group = group.sort_values('timestamp')
        # Alignment to 390 minutes (7:30 to 13:59)
        try:
            # Create 390-minute time range
            start_datetime = pd.to_datetime(f"{date} 07:30:00")
            end_datetime = pd.to_datetime(f"{date} 13:59:00")
            full_range = pd.date_range(start=start_datetime, end=end_datetime, freq='1min')
            
            # Remove duplicates, and index
            group_clean = group.drop_duplicates(subset=['timestamp'], keep='last')
            group_indexed = group_clean.set_index('timestamp')
            
            # Reindex to full trading day
            aligned_data = group_indexed.reindex(full_range)
            
            # bfill/ffill missing values for all columns
            aligned_data = aligned_data.ffill().bfill()
            
            # Reset index
            aligned_data = aligned_data.reset_index()
            aligned_data = aligned_data.rename(columns={'index': 'timestamp'})
            
            group = aligned_data
            
        except Exception as e:
            print(f"Error aligning {date}: {e}")
            continue
        
        # Truncate to max_length
        if max_sequence_length and len(group) > max_sequence_length:
            group = group.iloc[:max_sequence_length]
        
        # Check for missing information
        missing_features = [col for col in feature_cols if col not in group.columns]
        if missing_features:
            print(f"Warning: Missing features for {date}: {missing_features}")
            continue
        
        # Use ONLY the specified feature columns (NO price data like close)
        sequence = group[feature_cols].values
        
        # Handle NaN vals
        if np.isnan(sequence).any():
            nan_count = np.isnan(sequence).sum()
            total_values = sequence.size
            if nan_count / total_values > 0.1:  # Skip if >10% NaN as data bad
                print(f"Skipping {date}: too many NaN values ({nan_count}/{total_values})")
                continue
            # f/bfill NaN values to fix missing day data
            sequence = pd.DataFrame(sequence, columns=feature_cols).ffill().bfill().values
        
        # Calculate daily return (up/down) for labeling without close as feature
        first_close = group['close'].iloc[0]
        last_close = group['close'].iloc[-1]
        daily_return = (last_close - first_close) / first_close
        
        # Binary classification: "Up" (1.0) vs "Down" (2.0)
        label = 1.0 if daily_return > 0 else 2.0
        
        daily_sequences.append(sequence)
        labels.append(label)
        sequence_dates.append(date)
    
    print(f"Created {len(daily_sequences)} sequences")
    
    # Show class distribution for information
    up_days = labels.count(1.0)
    down_days = labels.count(2.0)
    total_days = len(labels)
    print(f"UP days: {up_days} ({up_days/total_days*100:.1f}%)")
    print(f"DOWN days: {down_days} ({down_days/total_days*100:.1f}%)")
    
    return daily_sequences, labels, sequence_dates

def generate_experiment_datasets(
    csv_filepath="original_datasets/1_min_SPY_2008-2021.csv",
    vix_filepath=None,  # "original_datasets/VIX_History.csv"
    macro_filepath=None,  # "original_datasets/macro_monthly.csv" 
    indicators_filepath="original_datasets/financial_indicators_full.csv",
    output_dir="./",
    test_size=0.3
):   
    
    os.makedirs(output_dir, exist_ok=True)
    
    #_df = dataframe
    print("====LOADING BASE DATA====")
    # Load S&P500
    price_df = load_sp500_data(csv_filepath)
    
    # Load other datasets
    vix_df = pd.DataFrame()
    if vix_filepath and os.path.exists(vix_filepath):
        vix_df = load_vix_data(vix_filepath)
        
    macro_df = pd.DataFrame()
    if macro_filepath and os.path.exists(macro_filepath):
        macro_df = load_macro_data(macro_filepath)
        
    indicators_df = pd.DataFrame()
    if indicators_filepath and os.path.exists(indicators_filepath):
        indicators_df = load_technical_indicators(indicators_filepath)
    
    experiments = []
    
    print("\n====EXPERIMENT 1: S&P500 ONLY====")
    df1 = price_df.copy()
    sequences1, labels1, dates1 = create_daily_sequences(df1)
    experiments.append({
        'name': 'SP500_ONLY',
        'sequences': sequences1,
        'labels': labels1,
        'dates': dates1,
        'features': ['open', 'high', 'low', 'close', 'volume', 'vwap']
    })
    
    if not vix_df.empty:
        print("\n====EXPERIMENT 2: S&P500 + VIX====")
        df2 = merge_vix_data(price_df.copy(), vix_df)
        sequences2, labels2, dates2 = create_daily_sequences(df2)
        experiments.append({
            'name': 'SP500_VIX',
            'sequences': sequences2,
            'labels': labels2,
            'dates': dates2,
            'features': ['open', 'high', 'low', 'close', 'volume', 'vwap', 'vix_open']
        })
    
    if not vix_df.empty and not macro_df.empty:
        print("\n====EXPERIMENT 3: S&P500 + VIX + MACRO====")
        df3 = merge_vix_data(price_df.copy(), vix_df)
        df3 = merge_macro_data(df3, macro_df, lag_months=2)
        sequences3, labels3, dates3 = create_daily_sequences(df3)
        macro_features = [col for col in macro_df.columns if col != 'DATE']
        experiments.append({
            'name': 'SP500_VIX_MACRO',
            'sequences': sequences3,
            'labels': labels3,
            'dates': dates3,
            'features': ['open', 'high', 'low', 'close', 'volume', 'vwap', 'vix_open'] + macro_features
        })
    
    if not vix_df.empty and not indicators_df.empty:
        print("\n====EXPERIMENT 4: S&P500 + VIX + TECHNICAL====")
        df4 = merge_vix_data(price_df.copy(), vix_df)
        df4 = merge_indicator_data(df4, indicators_df)
        sequences4, labels4, dates4 = create_daily_sequences(df4)
        tech_features = [col for col in indicators_df.columns if col != 'timestamp']
        experiments.append({
            'name': 'SP500_VIX_TECH',
            'sequences': sequences4,
            'labels': labels4,
            'dates': dates4,
            'features': ['open', 'high', 'low', 'close', 'volume', 'vwap', 'vix_open'] + tech_features
        })
    
    if not indicators_df.empty:
        print("\n====EXPERIMENT 5: TECHNICAL INDICATORS ONLY====")
        df5 = merge_indicator_data(price_df.copy(), indicators_df)
        
        # Keep only technicals (remove price data)
        tech_cols = [col for col in indicators_df.columns if col != 'timestamp']
        keep_cols = ['timestamp'] + tech_cols + ['close']  # Keep close for labeling
        df5_tech_only = df5[keep_cols].copy()
        
        sequences5, labels5, dates5 = create_daily_sequences_tech_only(df5_tech_only, tech_cols)
        experiments.append({
            'name': 'TECH_ONLY',
            'sequences': sequences5,
            'labels': labels5,
            'dates': dates5,
            'features': tech_cols
        })
    
    if not vix_df.empty and not macro_df.empty and not indicators_df.empty:
        print("\n====EXPERIMENT 6: ALL VARIABLES COMBINED====")
        df6 = merge_vix_data(price_df.copy(), vix_df)
        df6 = merge_macro_data(df6, macro_df, lag_months=2)
        df6 = merge_indicator_data(df6, indicators_df)
        sequences6, labels6, dates6 = create_daily_sequences(df6)
        all_features = (['open', 'high', 'low', 'close', 'volume', 'vwap', 'vix_open'] + 
                       macro_features + tech_features)
        experiments.append({
            'name': 'SP500_VIX_MACRO_TECH',
            'sequences': sequences6,
            'labels': labels6,
            'dates': dates6,
            'features': all_features
        })
    
    print("\n====GENERATING .TS FILES====")
    
    for exp in experiments:
        if not exp['sequences']:
            continue
            
        print(f"\nProcessing {exp['name']}...")
        
        # Pad sequences and prepare data
        X = uniform_sequences(exp['sequences'], max_length=390)
        y = np.array(exp['labels'])
        dates = np.array(exp['dates'])
        X = X.transpose(0, 2, 1)  # (samples, features, timesteps)
        
        # 70% train, 30% test
        sorted_indices = np.argsort(dates)
        split_idx = int(len(sorted_indices) * (1 - test_size))
        
        train_indices = sorted_indices[:split_idx]
        test_indices = sorted_indices[split_idx:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        print(f"  Data shape: {X.shape} (samples, features, timesteps)")
        print(f"  Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
        print(f"  Train split: {dates[train_indices[-1]]} | Test split: {dates[test_indices[0]]}")
        
        # New file's name
        train_file = os.path.join(output_dir, f"{exp['name']}_TRAIN.ts")
        test_file = os.path.join(output_dir, f"{exp['name']}_TEST.ts")
        
        # Write
        write_ts_file(train_file, X_train, y_train, exp['features'], exp['name'])
        write_ts_file(test_file, X_test, y_test, exp['features'], exp['name'])
        
        print(f"  Created: {train_file}")
        print(f"  Created: {test_file}")
    
    print(f"\nALL EXPERIMENT DATA GENERATION COMPLETED")
    print(f"Generated {len(experiments) * 2} .ts files in {output_dir}")
    
    return experiments

if __name__ == "__main__":
    # Configuration
    USE_VIX = True        
    USE_MACRO = True       
    USE_INDICATORS = True  
    
    # File paths
    csv_file = "original_datasets/1_min_SPY_2008-2021.csv"
    vix_file = "original_datasets/VIX_History.csv" if USE_VIX else None
    macro_file = "original_datasets/macro_monthly.csv" if USE_MACRO else None
    indicators_file = "original_datasets/financial_indicators_full.csv" if USE_INDICATORS else None
    
    print("DATA GENERATON FOR S&P500 EXPERIMENT")
    print(f"   S&P500 data: YES")
    print(f"   VIX data: {'YES' if USE_VIX else 'NO'}")
    print(f"   Macro indicators: {'YES' if USE_MACRO else 'NO'}")
    print(f"   Technical indicators: {'YES' if USE_INDICATORS else 'NO'}")
    print()
    
    experiments = generate_experiment_datasets(
        csv_filepath=csv_file,
        vix_filepath=vix_file,
        macro_filepath=macro_file,
        indicators_filepath=indicators_file,
        output_dir="./thesis_datasets/"
    )
