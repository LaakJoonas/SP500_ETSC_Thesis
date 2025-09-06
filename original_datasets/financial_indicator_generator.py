
#Financial technical indicators calculator

import pandas as pd
import numpy as np
from datetime import datetime

def rsi(data, period=14): #Some issues with getting data for some days. However, it works well enough.
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
#Using the SMA here as this is how it's defined in investopedia.


def macd(data, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = data.ewm(span=fast_period).mean()
    ema_slow = data.ewm(span=slow_period).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def cci(high, low, close, period=20):
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    return cci

def exponential_moving_average(data, period=20):
    return data.ewm(span=period).mean()

def calculate_all_indicators(df, save_file=True):
    
    #Calculate financial indicators from OHLCV
    
    #save_file: Whether to save results to CSV, for debug
    
    #return: dataframe with all indicators
    
    
    print("====CALCULATING FINANCIAL INDICATORS====")
    print(f"Input data shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Create a copy to avoid modifying original
    indicators_df = df.copy()
    
    # Sort by timestamp
    indicators_df = indicators_df.sort_values('timestamp')
    
    print("\nCalculating indicators...")
    
    # 1. EMA
    indicators_df['ema_26'] = exponential_moving_average(indicators_df['close'], 26)
    
    # 2. RSI
    indicators_df['rsi_14'] = rsi(indicators_df['close'], 14)
    
    # 3. MACD
    macd_line, signal_line, histogram = macd(indicators_df['close'])
    indicators_df['macd_histogram'] = histogram
    
    # 4. CCI
    indicators_df['cci'] = cci(indicators_df['high'], indicators_df['low'], indicators_df['close'], 20)
    
    # List indicator columns
    indicator_columns = [
        'ema_26',           # EMA
        'rsi_14',                                # RSI
        'macd_histogram',  # MACD
        'cci'                                    # CCI
    ]
    
    print(f"\nCalculated {len(indicator_columns)} indicators:")
    for i, indicator in enumerate(indicator_columns, 1):
        non_null_count = indicators_df[indicator].count()
        print(f"{i:2d}. {indicator:<15}: {non_null_count:,} valid values")
    
    # Add date column for merging
    indicators_df['date'] = indicators_df['timestamp'].dt.date
    
    if save_file:
        output_file = 'financial_indicators_full.csv'
        indicators_df.to_csv(output_file, index=False)
        print(f"\nSaved full data to: {output_file}")
        
        # save a daily summary (end-of-day values) for debug
        daily_indicators = indicators_df.groupby('date').last().reset_index()
        daily_file = 'financial_indicators_daily.csv'
        daily_indicators.to_csv(daily_file, index=False)
        print(f"Saved daily summary to: {daily_file}")
        
        # Show sample of results
        print(f"\nSample of calculated indicators (last 5 days):")
        sample_cols = ['date'] + indicator_columns[:4] 
        print(daily_indicators[sample_cols].tail().round(2))
    
    return indicators_df

def load_and_process_sp500_for_indicators():
    try:
        print("Loading SP500 data...")
        df = pd.read_csv('1_min_SPY_2008-2021.csv')
        print(f"Loaded {len(df):,} rows")
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['date'])
        
        # Keep necessary columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols].copy()
        
        # Remove rows with missing data
        df = df.dropna()
        
        print(f"After cleaning: {len(df):,} rows")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
        
    except FileNotFoundError:
        print("Error: 1_min_SPY_2008-2021.csv not found!")
        return None

if __name__ == "__main__":
    # Load SP500 and calculate indicators
    sp500_df = load_and_process_sp500_for_indicators()
    
    if sp500_df is not None: #debug
        # Calculate indicators
        indicators_df = calculate_all_indicators(sp500_df, save_file=True)
        
        print("\n" + "="*60)
        print("FINANCIAL INDICATORS CALCULATION COMPLETED")
        print(f"Created files:")
        print(f"   - financial_indicators_full.csv (minute-level data)")
        print(f"   - financial_indicators_daily.csv (daily summary, debug)")
    else:
        print("Failed to load SP500 data")
