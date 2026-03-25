# seq to seq with attention

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
'''
src = [[12, 33],    # 각 문장의 1번째 단어
       [45, 91],    # 각 문장의 2번째 단어
       [78, 56]]    # 각 문장의 3번째 단어

seq_len 축 (열, axis=1)
         단어1  단어2  단어3
문장1  [  12,    45,    78  ]   ← batch 축 (행, axis=0)
문장2  [  33,    91,    56  ]
#src: 인덱스 시퀀스 (batch, seq_len) 의 기준에 그렇다는 뜻
(seq_len, batch, emb_dim) 가 기본 lstm 입출력인데 batch first 해서 앞으로 뺌
'''
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)

        return outputs, hidden, cell

#바다나우 어텐션 (Bahdanau Attention) 구현

class BAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()

        self.W1 = nn.Linear(hid_dim, hid_dim)  #W1: decoder hidden state를 attention용 공간으로 변환
        self.W2 = nn.Linear(hid_dim, hid_dim) #W2: encoder outputs를 attention용 공간으로 변환
        self.v = nn.Linear(hid_dim, 1, bias=False) #context vector 연산 위한 파라미터

    def forward(self, hidden, encoder_outputs): 
        #hidden (batch_size, hidden_dim)
        #encoder_outputs (seq_len, batch_size, hidden_dim) 이 기본
        src_len = encoder_outputs.shape[0] #입력 길이 추출

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) #St-1 표현

        #원래 hidden 사이즈는 (batch_size, hidden_dim) 인데 중간에 차원 1 추가 후 -> src_len 길이만큼 복제
        #(batch_size, src_len, hidden_dim) 으로 변환
        encoder_outputs = encoder_outputs.permute(1,0,2) # 수식에서 h
        #(batch, src_len, hidden_dim) 위치별로  self.W1 와 self.W2 더해야 해서

        energy = torch.tanh(self.W1(hidden) + self.W2(encoder_outputs)) #e 계산 
        attention = self.v(energy).squeeze(2) # v 통과 하면 hidden_dim 이 제거됨. 마지막 차원 제거
        #batch, src_len
        #v가 hidden dim 이 1이 되어야 하는 이유는 스칼라 점수가 되어야 해서 즉 batch, src_len 으로
        #점수가 담겨져 있는거

        return nn.functional.softmax(attention, dim=1) #softmax 통과
        #dim=0 이면 세로, batch 끼리, 문장 끼리 비교
        #dim=1 은 가로, 즉 src_len 끼리, 문장 안에서 단어끼리 비교

#bmm -> 배치 기준으로 곱해라 라는 뜻
'''
permute() 함수란
permute()는 텐서의 차원(축) 순서를 재배치하는 함수입니다. 
데이터 자체는 바뀌지 않고, 축의 순서만 바꿉니다.

기존 shape: (축0, 축1, 축2)
                ↓     ↓     ↓
permute(1,0,2) → 새 축0=기존축1, 새 축1=기존축0, 새 축2=기존축2

즉, 축0과 축1을 서로 교환

이 어텐션은 linear layer 만든 후 연산한거라
다른 모달리티 연산시 필요

'''

