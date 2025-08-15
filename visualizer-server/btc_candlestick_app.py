
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import requests
import os
from db_func import get_db_connection, fetch_all_btc_four_six

# --- 페이지 및 환경 설정 ---
st.set_page_config(layout="wide")
st.title("모델별 예측 결과 시각화 대시보드")

# 환경 변수에서 서버 주소 가져오기 (없으면 기본값 사용)
INFERENCE_SERVER_URL = os.environ.get("INFERENCE_SERVER_URL", "http://localhost:8000")
REGISTERED_MODEL_NAME = "BTC_LSTM_Production"

# --- 사이드바 UI ---
st.sidebar.title("📈 모델 예측 제어")

@st.cache_data(ttl=60) # 1분마다 캐시 갱신
def get_model_aliases():
    """DB에서 직접 모델 별칭 목록을 가져옵니다."""
    try:
        # DB에서 직접 별칭을 조회하는 SQL 쿼리
        query = f"SELECT alias FROM registered_model_aliases WHERE name = '{REGISTERED_MODEL_NAME}' ORDER BY alias DESC"
        with get_db_connection() as conn:
            df = pd.read_sql_query(query, conn)
        
        aliases = df['alias'].tolist()

        if not aliases:
            st.sidebar.warning("등록된 모델 별칭이 없습니다.")
            return ["backtest_20250531"] # 샘플 데이터용 기본 별칭
        return aliases
    except Exception as e:
        st.sidebar.error(f"DB 연결 실패: {e}")
        # DB 연결 실패 시에도 샘플 별칭을 반환하여 앱이 멈추지 않도록 함
        return ["backtest_20250531"]

model_aliases = get_model_aliases()
model_aliases_prefix = list(map(lambda x: x.removeprefix('backtest_'), model_aliases))

selected_alias = st.sidebar.selectbox(
    "예측 기준 모델(Alias)을 선택하세요:",
    model_aliases_prefix,
    help="이 모델이 학습된 날짜 이후의 기간을 예측합니다."
)

predict_button = st.sidebar.button(
    "선택한 모델로 예측 생성",
    disabled=not selected_alias
)

clear_button = st.sidebar.button("예측 결과 모두 지우기")

st.sidebar.markdown("--- ")
st.sidebar.info(f"**추론 서버:** `{INFERENCE_SERVER_URL}`")

# --- 메인 대시보드 ---

# DB에서 데이터 가져오기
ohlcv_df = fetch_all_btc_four_six()
ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['datetime'])

# 예측 결과를 세션 상태에 저장하기 위한 초기화 (딕셔너리 형태)
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# 예측 결과 지우기 버튼 로직
if clear_button:
    st.session_state.predictions = {}

# 예측 생성 버튼 로직
if predict_button:
    with st.spinner(f"'{selected_alias}' 모델 기준으로 예측을 생성합니다..."):
        try:
            # 1. 별칭에서 'backtest_' 접두사를 제거하고 날짜 부분만 추출
            start_date_str = datetime.strptime(selected_alias, '%Y%m%d').strftime('%Y-%m-%d')
            
            # 2. 추론 서버에 예측 요청 (새로운 API 가상)
            api_endpoint = f"{INFERENCE_SERVER_URL}/predict_range"
            payload = {"start_date": start_date_str}
            response = requests.post(api_endpoint, json=payload, timeout=120)
            response.raise_for_status()
            
            pred_data = response.json()
            
            # 3. 결과 데이터프레임 생성
            pred_start_date = pd.to_datetime(pred_data['start_date'])
            pred_dates = [pred_start_date + timedelta(days=i) for i in range(len(pred_data['predictions']))]
            prediction_df = pd.DataFrame({
                'datetime': pred_dates,
                'prediction': pred_data['predictions']
            })
            # 딕셔너리에 현재 예측 결과 저장
            st.session_state.predictions[selected_alias] = prediction_df
            st.success(f"✅ '{selected_alias}' 모델 예측 성공!")

        except (requests.exceptions.RequestException, KeyError) as e:
            st.warning(f"API 호출 실패 ({e}). 샘플 예측 데이터를 표시합니다.")
            try:
                # 별칭에서 날짜 부분 추출
                date_part = selected_alias.replace("backtest_", "")
                prediction_start_date = datetime.strptime(date_part, '%Y%m%d') + timedelta(days=1)
                last_data_date = ohlcv_df['datetime'].max()
                if prediction_start_date <= last_data_date:
                    date_range = pd.date_range(start=prediction_start_date, end=last_data_date)
                    last_close_price = ohlcv_df[ohlcv_df['datetime'] < prediction_start_date]['btc_close'].iloc[-1]
                    sample_predictions = last_close_price + np.random.randn(len(date_range)).cumsum() * 50
                    prediction_df = pd.DataFrame({
                        'datetime': date_range,
                        'prediction': sample_predictions
                    })
                    # 딕셔너리에 현재 예측 결과 저장
                    st.session_state.predictions[selected_alias] = prediction_df
                else:
                    st.error("예측 시작일이 데이터 기간을 벗어납니다.")
            except (IndexError, ValueError):
                 st.error("샘플 데이터를 생성할 기준 날짜를 찾지 못했습니다.")

# --- 차트 그리기 ---
fig = go.Figure()

# 1. 실제 가격 캔들스틱 차트
fig.add_trace(go.Candlestick(x=ohlcv_df['datetime'],
                             open=ohlcv_df['btc_open'],
                             high=ohlcv_df['btc_high'],
                             low=ohlcv_df['btc_low'],
                             close=ohlcv_df['btc_close'],
                             name='실제 가격'))

# 2. 저장된 모든 예측값을 순회하며 라인 차트로 추가
if st.session_state.predictions:
    # 색상 리스트 준비
    colors = ['orange', 'purple', 'green', 'cyan', 'magenta', 'yellow']
    color_idx = 0
    for alias, pred_df in st.session_state.predictions.items():
        fig.add_trace(go.Scatter(
            x=pred_df['datetime'],
            y=pred_df['prediction'],
            mode='lines',
            line=dict(color=colors[color_idx % len(colors)], width=3),
            name=f'{alias} 모델 예측'
        ))
        color_idx += 1

# 차트 레이아웃
min_price = ohlcv_df['btc_low'].min()
max_price = ohlcv_df['btc_high'].max()
padding = (max_price - min_price) * 0.05

fig.update_layout(
    title='BTC/USD Candlestick Chart & Model Prediction',
    yaxis_title='Price (USD)',
    xaxis_title='Date',
    xaxis_rangeslider_visible=False,
    yaxis_range=[min_price - padding, max_price + padding],
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# 데이터 테이블
st.subheader("원본 데이터 미리보기")
st.dataframe(ohlcv_df.tail())
