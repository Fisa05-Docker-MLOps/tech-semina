import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDatasetXY(Dataset):
    """
    사전 분리된 입력(X)과 타겟(y) 데이터를 받아
    LSTM 모델 학습을 위한 시계열 데이터셋을 생성합니다.

    Args:
        x_data (np.ndarray): 모델의 입력 특성 데이터 (샘플 수, 피처 수).
        y_data (np.ndarray): 모델의 타겟(정답) 데이터 (샘플 수, 1).
        sequence_length (int): 모델에 입력으로 사용할 시퀀스의 길이.
    """
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, sequence_length: int):
        self.sequence_length = sequence_length
        # 데이터를 PyTorch 텐서로 변환
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        
    def __len__(self) -> int:
        # 생성 가능한 총 샘플의 개수는 동일한 원리로 계산됩니다.
        return len(self.x_data) - self.sequence_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        주어진 인덱스(idx)에 해당하는 (입력 시퀀스, 타겟) 쌍을 반환합니다.
        """
        # 입력 시퀀스 X: x_data에서 idx부터 (idx + sequence_length)까지 슬라이싱
        x = self.x_data[idx : idx + self.sequence_length]

        # 타겟 y: y_data에서 입력 시퀀스 바로 다음 시점의 값을 가져옴
        y = self.y_data[idx + self.sequence_length]
        
        return x, y
    

class TimeSeriesDatasetX(Dataset):
    """
    사전 분리된 입력(X) 데이터만을 받아
    LSTM 모델 예측을 위한 시계열 데이터셋을 생성합니다.

    Args:
        x_data (np.ndarray): 모델에 입력할 특성 데이터 (샘플 수, 피처 수).
        sequence_length (int): 모델에 입력으로 사용할 시퀀스의 길이.
    """
    def __init__(self, x_data: np.ndarray, sequence_length: int):
        self.sequence_length = sequence_length
        # 데이터를 PyTorch 텐서로 변환
        self.x_data = torch.FloatTensor(x_data)

    def __len__(self) -> int:
        """
        생성 가능한 총 시퀀스의 개수를 반환합니다.
        """
        # 전체 데이터 길이에서 시퀀스 길이를 빼면 생성 가능한 샘플 수가 됩니다.
        return len(self.x_data) - self.sequence_length + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        주어진 인덱스(idx)에 해당하는 입력 시퀀스를 반환합니다.
        """
        # 입력 시퀀스 X: x_data에서 idx부터 (idx + sequence_length)까지 슬라이싱
        x = self.x_data[idx : idx + self.sequence_length]
        return x