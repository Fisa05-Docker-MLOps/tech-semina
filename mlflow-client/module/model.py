from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTM, self).__init__()
        # Initialize parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define LSTM layer and fully connected layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> torch.Tensor:
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # take the last time step
        return out
    

def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    weights_path: Optional[str] = None,
    clip: Optional[float] = None
) -> float:
    """
    모델을 한 에폭(epoch) 동안 학습시키고 평균 손실을 반환합니다.

    Args:
        model (nn.Module): 학습할 PyTorch 모델.
        data_loader (DataLoader): 학습 데이터로더.
        criterion (nn.Module): 손실 함수 (예: nn.MSELoss).
        optimizer (torch.optim.Optimizer): 옵티마이저 (예: torch.optim.Adam).
        device (torch.device): 학습에 사용할 디바이스 ('cuda' 또는 'cpu').
        weights_path (Optional[str], optional): 불러올 모델 가중치 파일 경로. 기본값은 None.
        clip (Optional[float], optional): 그래디언트 클리핑(gradient clipping) 임계값. 
                                           RNN 계열 모델의 안정적인 학습에 도움을 줍니다. 기본값은 None.

    Returns:
        Tuple[float, float]: 에폭의 평균 손실(average loss)과 평균 정확도(average accuracy).
    """
    # 저장된 가중치가 있다면 불러오기
    if weights_path:
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"✅ '{weights_path}' 에서 모델 가중치를 성공적으로 불러왔습니다.")
        except FileNotFoundError:
            print(f"⚠️ 경고: '{weights_path}' 경로에 파일이 없어 가중치를 불러오지 못했습니다.")
        except Exception as e:
            print(f"🚨 오류: 가중치 로딩 중 문제 발생 - {e}")

    model.train()  # 모델을 학습 모드로 설정

    epoch_loss = 0.0
    total_samples = 0

    # 데이터로더를 순회하며 미니배치 단위로 학습 진행
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

       # 회귀에서는 target의 shape을 모델 출력과 맞춰주어야 할 수 있음
        # 예: target이 [batch_size]일 경우, [batch_size, 1]로 변경
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        # 순전파 (Forward pass)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 역전파 및 최적화 (Backward pass and optimization)
        optimizer.zero_grad()
        loss.backward()

        # 그래디언트 클리핑 (Gradient Clipping)
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
        optimizer.step()

        # 에폭 전체의 손실 집계
        epoch_loss += loss.item() * inputs.size(0)
        total_samples += targets.size(0)
        
        # 진행 상황 로깅
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(data_loader)} | Batch Loss: {loss.item():.4f}")

    # 에폭의 평균 손실 계산
    avg_epoch_loss = epoch_loss / total_samples
    
    return avg_epoch_loss


def predict_single(
    model: nn.Module,
    input_sequence: torch.Tensor,
    device: torch.device
) -> float:
    """
    단일 입력 시퀀스에 대한 회귀 예측을 수행합니다.

    Args:
        model (nn.Module): 학습이 완료된 모델.
        input_sequence (torch.Tensor): 예측에 사용할 단일 입력 시퀀스 데이터.
                                       형태: (sequence_length, input_size)
        device (torch.device): 연산을 수행할 디바이스.

    Returns:
        float: 모델의 단일 예측값.
    """
    model.eval()
    
    # 1. 입력 데이터를 디바이스로 이동
    input_tensor = input_sequence.to(device)
    
    # 2. 모델 입력에 맞게 배치 차원(batch dimension) 추가
    # 모델은 (batch_size, sequence_length, input_size) 형태를 기대하므로,
    # (sequence_length, input_size) -> (1, sequence_length, input_size)로 변경
    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor)
        
    # 3. 결과를 CPU로 이동시키고 스칼라 값으로 변환하여 반환
    return prediction.cpu().item()


def predict_batch(
    model: nn.Module,
    batch_input: torch.Tensor,
    device: torch.device
) -> np.ndarray:
    """
    단일 입력 배치에 대한 모델의 회귀 예측을 수행하고, 결과를 NumPy 배열로 반환합니다.

    Args:
        model (nn.Module): 학습이 완료된 모델.
        batch_input (torch.Tensor): 예측할 입력 데이터 배치.
        device (torch.device): 연산을 수행할 디바이스.

    Returns:
        np.ndarray: 모델의 예측값을 담은 NumPy 배열.
    """
    model.eval()  # 모델을 평가 모드로 설정

    with torch.no_grad():  # 경사도 계산 비활성화
        predictions = model(batch_input.to(device))
    
    # 예측값을 CPU로 이동시키고 NumPy 배열로 변환
    predictions = predictions.cpu().numpy()

    return predictions


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> np.ndarray:
    """
    전체 데이터로더에 대해 회귀 예측을 수행하고, 
    모든 예측값을 하나의 NumPy 배열로 반환합니다.

    Args:
        model (nn.Module): 학습이 완료된 모델.
        data_loader (DataLoader): 예측을 수행할 전체 데이터셋의 데이터로더.
                               이 데이터로더는 입력 텐서(X)만을 반환해야 합니다.
        device (torch.device): 연산을 수행할 디바이스.

    Returns:
        np.ndarray: 전체 데이터셋에 대한 예측값들을 담은 단일 NumPy 배열.
    """
    model.eval()
    all_predictions = []

    # !!!!!!!!!! 데이터로더가 입력 텐서(inputs)만 반환하도록 수정 필요
    for inputs, _ in data_loader:
        # predict_batch는 배치의 예측 결과를 np.ndarray로 반환
        batch_predictions_np = predict_batch(model, inputs, device)
        all_predictions.append(batch_predictions_np)
    
    # NumPy 배열들의 리스트를 하나의 큰 배열로 결합
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    return all_predictions