
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta, date
import requests
from db_func import fetch_all_btc

st.set_page_config(layout="wide")

st.title("비트코인(BTC) 캔들차트 시각화")

# API로부터 OHLCV 데이터 가져오는 함수
# @st.cache_data(ttl=3600) # 1시간 캐시
# def fetch_btc_data_from_api(target_date: date):
#     base_url = "http://localhost:8000" # 추론 서버 주소
#     fetch_endpoint = f"{base_url}/fetch"
    
#     params = {"end_date": target_date.strftime("%Y-%m-%d")}
    
#     try:
#         response = requests.get(fetch_endpoint, params=params)
#         response.raise_for_status() # HTTP 오류 발생 시 예외 발생
#         data = response.json()
        
#         if data.get("status") == "success":
#             df = pd.DataFrame(data["data"])
#             df['Date'] = pd.to_datetime(df['Date'])
#             return df
#         else:
#             st.error(f"API 응답 오류: {data.get("detail", "알 수 없는 오류")}")
#             return pd.DataFrame()
#     except requests.exceptions.ConnectionError:
#         st.error("API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요 (http://localhost:8000).")
#         return pd.DataFrame()
#     except requests.exceptions.RequestException as e:
#         st.error(f"API 요청 중 오류 발생: {e}")
#         return pd.DataFrame()

# 날짜 선택 위젯
selected_date = st.sidebar.date_input("데이터 기준 날짜 선택", datetime.now().date())

# 데이터 가져오기
ohlcv_df = pd.DataFrame()
# try:
#     ohlcv_df = fetch_btc_data_from_api(selected_date)
# except:
    # db 에서 가져오기
ohlcv_df = fetch_all_btc()

if not ohlcv_df.empty:
    # 캔들차트 생성
    fig = go.Figure(data=[go.Candlestick(x=ohlcv_df['datetime'],
                    open=ohlcv_df['btc_open'],
                    high=ohlcv_df['btc_high'],
                    low=ohlcv_df['btc_low'],
                    close=ohlcv_df['btc_close'])])

    fig.update_layout(
        title='BTC/USD Candlestick Chart with SMA 60',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False # 하단 범위 슬라이더 숨기기
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("데이터 미리보기")
    st.dataframe(ohlcv_df.tail())

    st.subheader("BTC 종가 데이터")
    st.dataframe(ohlcv_df[['datetime', 'btc_close']].tail())
