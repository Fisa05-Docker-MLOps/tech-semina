import os
import random
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def seed_everything(seed: int = 42) -> None:
    """
    모든 라이브러리의 시드를 설정하여 재현 가능한 결과를 보장합니다.
    
    Args:
        seed (int): 설정할 시드 값. 기본값은 42입니다.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


def generate_date_list(start_date_str, end_date_str, interval_days):
    """
    주어진 시작일, 종료일, 간격에 따라 날짜 문자열 리스트를 생성합니다.

    :param start_date_str: 시작 날짜 (YYYY-MM-DD 형식의 문자열)
    :param end_date_str: 종료 날짜 (YYYY-MM-DD 형식의 문자열)
    :param interval_days: 날짜 간격 (일)
    :return: 날짜 문자열 리스트
    """
    
    # 시작일과 종료일을 datetime 객체로 변환
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    
    # 날짜 간격을 timedelta 객체로 변환
    delta = datetime.timedelta(days=interval_days)
    
    date_list = []
    current_date = start_date
    
    # 현재 날짜가 종료 날짜보다 작거나 같을 때까지 반복
    while current_date <= end_date:
        # 날짜를 'YYYY-MM-DD' 형식의 문자열로 변환하여 리스트에 추가
        date_list.append(current_date.strftime("%Y-%m-%d"))
        # 현재 날짜에 간격을 더함
        current_date += delta
        
    return date_list