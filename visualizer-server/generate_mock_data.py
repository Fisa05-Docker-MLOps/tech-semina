import json
import pandas as pd
import numpy as np

# 1. 시간 인덱스 생성 (1시간 간격)
date_range = pd.date_range(start="2024-02-01 00:00:00", end="2025-06-30 23:00:00", freq="H")

# 2. 샘플 BTC 종가 생성 (임의값, 랜덤 변동)
np.random.seed(42)  # 재현 가능
base_price = 34500  # 시작 가격
prices = base_price + np.cumsum(np.random.randn(len(date_range)) * 50)  # 시간별 변동

# 3. 날짜 문자열 포맷 변경
date_strs = date_range.strftime("%Y.%m.%d %H:%M:%S")

# 4. JSON용 딕셔너리 생성
btc_data = {date: float(f"{price:.2f}") for date, price in zip(date_strs, prices)}

# 5. JSON 파일로 저장
with open("btc_sample_forecast.json", "w") as f:
    json.dump(btc_data, f, indent=2)

print("btc_sample_forecast.json 파일이 생성되었습니다.")
