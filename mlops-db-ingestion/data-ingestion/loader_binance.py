import time
import pandas as pd
import ccxt
from datetime import datetime

def fetch_btc_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    exchange = ccxt.binanceus()
    symbol = 'BTC/USDT'
    timeframe = '1h'
    limit = 1000
    since = int(start_date.timestamp() * 1000)

    all_data = []

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            break

        all_data.extend(ohlcv)
        last_ts = ohlcv[-1][0] + 1
        since = last_ts

        if last_ts > int(end_date.timestamp() * 1000):
            break

        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df = df[df['datetime'] <= end_date]
    return df
