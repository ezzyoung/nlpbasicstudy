import torch
import torch.nn as nn


class RNNCell(nn.Module):
    """RNN 셀을 직접 구현 (nn.RNN을 쓰지 않고 순환 구조를 수동으로 구성)"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x_t, h_prev):
        # h_t = tanh(W_ih * x_t + W_hh * h_prev + b)
        h_t = self.activation(self.i2h(x_t) + self.h2h(h_prev))
        return h_t


class CustomRNN(nn.Module):
    """
    커스텀 RNN 모델
    - 입력층: Linear (input_size -> hidden_size)
    - 은닉층: RNNCell (시간 축을 따라 순환)
    - 출력층: Linear (hidden_size -> output_size)
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.rnn_cells = nn.ModuleList(
            [RNNCell(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def _init_hidden(self, batch_size, device):
        return [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]

    def forward(self, x, hidden=None):
        """
        x: (batch_size, seq_len, input_size)
        반환: output (batch_size, output_size), hidden states
        """
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)

        # 입력층: 각 타임스텝의 입력을 hidden_size 차원으로 변환
        x = self.input_layer(x)  # (batch, seq_len, hidden_size)

        # 은닉층: 시간 축을 따라 순환 연산 수행
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, hidden_size)
            for layer_idx, rnn_cell in enumerate(self.rnn_cells):
                x_t = rnn_cell(x_t, hidden[layer_idx])
                hidden[layer_idx] = x_t

        # 출력층: 마지막 타임스텝의 은닉 상태로 최종 출력 생성
        output = self.output_layer(hidden[-1])  # (batch, output_size)

        return output, hidden


# 데모: 사인파 예측 (다음 값 예측)
if __name__ == "__main__":
    import math

    torch.manual_seed(42)

    # 하이퍼파라미터
    INPUT_SIZE = 1
    HIDDEN_SIZE = 32
    OUTPUT_SIZE = 1
    NUM_LAYERS = 2
    SEQ_LEN = 20
    EPOCHS = 300
    LR = 0.005

    # 사인파 데이터 생성
    def generate_sine_data(n_samples=200, seq_len=20):
        X, Y = [], []
        for i in range(n_samples):
            start = i * 0.1
            seq = [math.sin(start + j * 0.1) for j in range(seq_len + 1)]
            X.append(seq[:-1])
            Y.append(seq[-1])
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (n, seq_len, 1)
        Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)  # (n, 1)
        return X, Y

    X, Y = generate_sine_data()
    print(f"데이터 크기 - X: {X.shape}, Y: {Y.shape}")

    # 모델 생성
    model = CustomRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
    print(f"\n모델 구조:\n{model}\n")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total_params:,}\n")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 학습
    for epoch in range(1, EPOCHS + 1):
        model.train()
        output, _ = model(X)
        loss = criterion(output, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"[Epoch {epoch:>3d}/{EPOCHS}]  Loss: {loss.item():.6f}")

    # 예측 결과 확인
    model.eval()
    with torch.no_grad():
        pred, _ = model(X[:5])
        print("\n-- 예측 vs 정답 (앞 5개) --")
        for i in range(5):
            print(f"  예측: {pred[i].item():>8.4f}  |  정답: {Y[i].item():>8.4f}")
