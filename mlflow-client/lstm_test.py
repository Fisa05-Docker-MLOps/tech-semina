import torch
import numpy as np
import pandas as pd
import sklearn
import mlflow

#!/usr/bin/env python3
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from module.utils import seed_everything
from module.model import LSTM, train_epoch, predict_single, predict_batch, predict

def print_versions():
    print("PyTorch version:", torch.__version__)
    print("NumPy version:", np.__version__)
    print("Pandas version:", pd.__version__)
    print("scikit-learn version:", sklearn.__version__)
    print("MLflow version:", mlflow.__version__)

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

def main():
    # Print version information for dependencies
    print_versions()

    # Check for GPU availability
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # Define model parameters
    num_samples = 5000  # number of samples
    input_size = 18   # number of features
    hidden_size = 32
    num_layers = 2
    output_size = 1

    # Set random seed for reproducibility
    seed_everything(42)

    # Initialize the LSTM model and move it to the appropriate device
    model = LSTM(input_size, hidden_size, num_layers, output_size).to(DEVICE)
    model.eval()  # set model to evaluation mode

    # Create dummy input data: (batch_size, seq_length, input_size)
    dummy_input = torch.randn(num_samples, input_size, device=DEVICE)
    dummy_target = torch.randn(num_samples, 1, device=DEVICE)

    # Training parameters
    num_epochs = 10
    learning_rate = 0.001

    # Switch model to training mode
    model.train()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batch_size = 64
    seq_length = 12
    # Define train data loader (dummy for this example)
    train_dataset = TimeSeriesDatasetXY(x_data=dummy_input.cpu().numpy(), y_data=dummy_target.cpu().numpy(), sequence_length=seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        # Zero the gradients
        optimizer.zero_grad()
        
        loss = train_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            weights_path=None,  # No pre-trained weights
            clip=None  # No gradient clipping
        )
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")

    # After training, switch back to evaluation mode for inference
    model.eval()

    # 단일 샘플 예측
    single_input = torch.randn(seq_length, input_size, device=DEVICE)
    output = predict_single(model, single_input, DEVICE)

    # 배치 단위 예측
    for x, y in train_loader:
        output = predict_batch(model, x, DEVICE)

    output_all = predict(model, train_loader, DEVICE)

    # Detach model parameters (weights) and store them in a dictionary
    detached_weights = {}
    for name, param in model.named_parameters():
        detached_weights[name] = param.detach().cpu()
        print(f"Detached weights for {name}: {detached_weights[name].shape}")

if __name__ == "__main__":
    main()