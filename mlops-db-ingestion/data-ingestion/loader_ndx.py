import pandas as pd
import yfinance as yf
from sqlalchemy.engine import Connection

def fetch_ndx_data(start_date: str, conn: Connection) -> pd.DataFrame:
    """
    yfinance에서 ndx 데이터를 가져오되, 데이터가 없으면 DB의 마지막 값으로 대체
    """
    ticker = 'QQQ'
    try:
        df = yf.download(ticker, start=start_date, interval='1h', prepost=True)
        if df.empty:
            last_df = pd.read_sql(
                "SELECT * FROM integrated_data WHERE ndx_close IS NOT NULL ORDER BY datetime DESC LIMIT 1", conn
            )
            if last_df.empty:
                print(f"{ticker} 데이터 없음, DB에도 마지막 값 없음")
                return pd.DataFrame()
            
            # ndx 관련 컬럼만 가져옴
            ndx_cols = ['datetime','ndx_open','ndx_high','ndx_low','ndx_close']

            # BTC 데이터 timestamp에 맞춰 마지막 값 복사
            last_row = last_df.iloc[0][ndx_cols].copy()
            last_row['datetime'] = pd.to_datetime(start_date)
            return pd.DataFrame([last_row])
        
        # 정상적으로 데이터 받아온 경우
        df.reset_index(inplace=True)
        df = df[['Datetime','Open','High','Low','Close']]  # 데이터 고르기
        df.columns = ['datetime', 'open', 'high', 'low', 'close']
        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_convert(None)
        return df
    except Exception as e:
        print(f"NDX 데이터 수집 중 오류 발생: {e}")
        return pd.DataFrame()
