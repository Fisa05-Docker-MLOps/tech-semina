import pandas as pd
import yfinance as yf

def fetch_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    ticker = '^VIX'
    df = yf.download(ticker, start=start_date, end=end_date, interval='1h')
    df.reset_index(inplace=True)
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df = df[['datetime', 'open', 'high', 'low', 'close']]  # volume 제거
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_convert(None)
    return df
