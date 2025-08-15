
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import requests
import mlflow
import os
from db_func import fetch_all_btc_four_six

# --- 페이지 및 환경 설정 ---
st.set_page_config(layout="wide")
st.title("모델별 예측 결과 시각화 대시보드")

# 환경 변수에서 서버 주소 가져오기 (없으면 기본값 사용)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
INFERENCE_SERVER_URL = os.environ.get("INFERENCE_SERVER_URL", "http://localhost:8000")
REGISTERED_MODEL_NAME = "BTC_LSTM_Production"

# --- 사이드바 UI ---
st.sidebar.title("📈 모델 예측 제어")

@st.cache_data(ttl=60) # 1분마다 캐시 갱신
def get_model_aliases():
    """MLflow에서 모델 별칭 목록을 가져옵니다."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_registered_model(REGISTERED_MODEL_NAME).latest_versions
        aliases = set()
        for v in versions:
            for alias in v.aliases:
                aliases.add(alias)
        if not aliases:
            st.sidebar.warning("등록된 모델 별칭이 없습니다.")
            return ["20250531"] # 샘플 데이터용 기본 별칭
        return sorted(list(aliases), reverse=True)
    except Exception:
        st.sidebar.error("MLflow 서버에 연결할 수 없습니다.")
        return ["20250531"] # 샘플 데이터용 기본 별칭

model_aliases = get_model_aliases()
selected_alias = st.sidebar.selectbox(
    "예측 기준 모델(Alias)을 선택하세요:",
    model_aliases,
    help="이 모델이 학습된 날짜 이후의 기간을 예측합니다."
)

predict_button = st.sidebar.button(
    "선택한 모델로 예측 생성",
    disabled=not selected_alias
)

st.sidebar.markdown("--- ")
st.sidebar.info(f"**MLflow 서버:** `{MLFLOW_TRACKING_URI}`")
st.sidebar.info(f"**추론 서버:** `{INFERENCE_SERVER_URL}`")

# --- 메인 대시보드 ---

# DB에서 데이터 가져오기
ohlcv_df = fetch_all_btc_four_six()
ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['datetime'])

# 예측 결과를 세션 상태에 저장하기 위한 초기화
if 'prediction_df' not in st.session_state:
    st.session_state.prediction_df = None

# 예측 버튼 로직
if predict_button:
    with st.spinner(f"'{selected_alias}' 모델 기준으로 예측을 생성합니다..."):
        try:
            # 1. 별칭에서 시작 날짜 파싱
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
            st.session_state.prediction_df = pd.DataFrame({
                'datetime': pred_dates,
                'prediction': pred_data['predictions']
            })
            st.success("✅ 예측 성공!")

        except (requests.exceptions.RequestException, KeyError) as e:
            st.warning(f"API 호출 실패 ({e}). 샘플 예측 데이터를 표시합니다.")
            # --- 샘플 데이터 생성 로직 ---
            try:
                prediction_start_date = datetime.strptime(selected_alias, '%Y%m%d') + timedelta(days=1)
                last_data_date = ohlcv_df['datetime'].max()
                if prediction_start_date > last_data_date:
                    st.error("예측 시작일이 데이터 기간을 벗어납니다.")
                    st.session_state.prediction_df = None
                else:
                    date_range = pd.date_range(start=prediction_start_date, end=last_data_date)
                    last_close_price = ohlcv_df[ohlcv_df['datetime'] < prediction_start_date]['btc_close'].iloc[-1]
                    # 간단한 노이즈를 추가한 샘플 예측값 생성
                    sample_predictions = last_close_price + np.random.randn(len(date_range)).cumsum() * 50
                    st.session_state.prediction_df = pd.DataFrame({
                        'datetime': date_range,
                        'prediction': sample_predictions
                    })
            except (IndexError, ValueError):
                 st.error("샘플 데이터를 생성할 기준 날짜를 찾지 못했습니다.")
                 st.session_state.prediction_df = None

# --- 차트 그리기 ---
fig = go.Figure()

# 1. 실제 가격 캔들스틱 차트
fig.add_trace(go.Candlestick(x=ohlcv_df['datetime'],
                             open=ohlcv_df['btc_open'],
                             high=ohlcv_df['btc_high'],
                             low=ohlcv_df['btc_low'],
                             close=ohlcv_df['btc_close'],
                             name='실제 가격'))

# 2. 예측값이 있으면 라인 차트로 추가
if st.session_state.prediction_df is not None:
    pred_df = st.session_state.prediction_df
    fig.add_trace(go.Scatter(
        x=pred_df['datetime'],
        y=pred_df['prediction'],
        mode='lines',
        line=dict(color='orange', width=3),
        name=f'{selected_alias} 모델 예측'
    ))

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
