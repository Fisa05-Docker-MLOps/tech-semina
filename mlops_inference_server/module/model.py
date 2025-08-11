import torch
from torch import nn

class LSTM(nn.Module):
    """
    학습 스크립트와 동일한 시그니처/동작을 가진 LSTM 회귀 모델.
    입력: x [B, T, F]
    출력: y [B, 1] (마지막 타임스텝의 hidden을 FC로 매핑)
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))      # out: [B, T, H]
        out = self.fc(out[:, -1, :])         # [B, 1]
        return out
