from module.utils import generate_date_list
from module.mlflow_train import train_model_with_month


if __name__ == "__main__":
    # 백테스팅할 날짜 목록 생성
    start = "2024-02-01"
    end = "2025-06-30"
    interval = 7
    backtest_dates = generate_date_list(start, end, interval)

    for date_str in backtest_dates:
        print(f"\n--- {date_str} 기준 모델 학습 시작 ---")
        
        alias = f"backtest_{date_str.replace('-', '')}"
        train_model_with_month(base_date=date_str, alias=alias)