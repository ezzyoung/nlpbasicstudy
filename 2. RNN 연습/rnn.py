import torch
import torch.nn as nn


# RNN Cell 구현
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)  # 입력층 -> hidden 층
        self.h2h = nn.Linear(hidden_size, hidden_size)  # hidden 층 -> hidden 층
        self.activation = nn.Tanh()  # 활성화 함수

    def forward(self, x_t, h_prev):
        # h_t = tanh(W_ih * x_t + W_hh * h_prev + b)
        h_t = self.activation(self.i2h(x_t) + self.h2h(h_prev))
        return h_t


class CustomRNN(nn.Module):
    """
    커스텀 RNN 모델
    - 입력층: Linear (input_size -> hidden_size)
    - 은닉층: RNNCell (num_layers 만큼 쌓아서 순환)
    - 출력층: Linear (hidden_size -> output_size)
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size #다른 메서드에서 쓰기 위해 저장
        self.num_layers = num_layers

        # 입력층
        self.input_layer = nn.Linear(input_size, hidden_size)

        # 은닉층
        self.rnn_cells = nn.ModuleList(
            [RNNCell(hidden_size, hidden_size) for _ in range(num_layers)]
        ) # 여러개의 nn.Module 를 담아두고 해석 가능한 리스트

        # 출력층
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_prev=None):
        """
        Args:
            x: 입력 텐서 (batch, seq_len, input_size)
            h_prev: 초기 은닉 상태 (num_layers, batch, hidden_size) 또는 None

        Returns:
            outputs: 전체 시퀀스 출력 (batch, seq_len, output_size)
            h_t: 마지막 타임스텝의 은닉 상태 (num_layers, batch, hidden_size)
        """
        batch_size, seq_len, _ = x.size() #입력 텐서에서 정보 빼기

        # 초기 은닉 상태가 없으면 0으로 초기화
        if h_prev is None:
            h_prev = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=x.device
            ) #0 초기화, x.device 는 gpu 인지 cpu 인지 확인하고 그에 맞게 메모리 설정 
            # 예를 들어 (2,4,16) -> layer 2개, batch 4개, hidden size 16

        # 각 레이어의 은닉 상태를 리스트로 분리
        h_states = [h_prev[layer] for layer in range(self.num_layers)] #h_prev 를 레이어별로 분리 즉 h_states[0] = h_prev[0], h_states[1] = h_prev[1] (4,16)

        outputs = []
        for t in range(seq_len): #seq_len, 즉 타임스텝 만큼 반복
            # 입력층 통과
            layer_input = self.input_layer(x[:, t, :])  # (batch, hidden_size) 즉 t 정수 인덱싱이 가운데 사라지게 함

            # 각 레이어의 RNNCell을 순서대로 통과
            for layer in range(self.num_layers):
                h_states[layer] = self.rnn_cells[layer](layer_input, h_states[layer])
                layer_input = h_states[layer]  # 현재 레이어 출력 = 다음 레이어 입력

            # 마지막 레이어의 출력을 output_layer에 전달
            outputs.append(self.output_layer(h_states[-1]))

        # (seq_len, batch, output_size) -> (batch, seq_len, output_size)
        outputs = torch.stack(outputs, dim=1) #seq_len 기준으로 쌓아야 하니까

        # 은닉 상태를 다시 (num_layers, batch, hidden_size)로 합침
        h_t = torch.stack(h_states, dim=0) #레이어 기준으로 쌓는다. nn.RNN 이 은닉 상태를 (num_layers, batch, hidden_size) 로 반환하기 때문에 지정

        return outputs, h_t
