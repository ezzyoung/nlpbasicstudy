import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from rnn import CustomRNN

'''
부모 자식 클래스 정의

자식에서 오버라이드한 메서드는 rnn_run.py 즉 자식의 것을 따름
그러나 자식에서 정의 안한 메서드는 rnn.py 인 부모, 즉 상속한 것을 따름

'''

# 데이터로드
train_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
test_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"

train_df = pd.read_csv(train_url, sep='\t').dropna(subset=['document'])
test_df = pd.read_csv(test_url, sep='\t').dropna(subset=['document'])

print(f"학습 데이터: {len(train_df)}건, 검증 데이터: {len(test_df)}건")


# 어휘사전 구축 및 토큰화
def build_vocab(texts, min_freq=2):
    counter = Counter() #요소의 등장 횟수를 세는 딕셔너리 Counter({'a': 3, 'b': 2, 'c': 1})
    for text in texts:
        counter.update(list(str(text))) #횟수 누적
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for char, freq in counter.most_common():
        if freq >= min_freq:
            vocab[char] = len(vocab)
    return vocab


vocab = build_vocab(train_df['document'])
print(f"어휘 사전 크기: {len(vocab)}")


# 텍스트 -> 인덱스 시퀀스 반환
MAX_LEN = 50


def encode_text(text, vocab, max_len=MAX_LEN):
    text = str(text)
    indices = [vocab.get(c, vocab['<UNK>']) for c in text[:max_len]]
    indices += [vocab['<PAD>']] * (max_len - len(indices))
    return indices


# 데이터셋 클래스: 원시 데이터를 모델이 이해할 수 있는 텐서로 변환
class NSMCDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_LEN):
        self.encoded = [encode_text(t, vocab, max_len) for t in texts]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# 데이터로더 설정
train_dataset = NSMCDataset(train_df['document'], train_df['label'], vocab)
val_dataset = NSMCDataset(test_df['document'], test_df['label'], vocab)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"train_loader 배치 수: {len(train_loader)}, val_loader 배치 수: {len(val_loader)}")


#감정 분류용으로 custom 모델
class SentimentRNN(nn.Module):
    """
    Embedding → CustomRNN → 마지막 타임스텝 출력으로 분류
    - Embedding: 정수 인덱스 → 밀집 벡터 (vocab_size → embed_dim)
    - CustomRNN: 시퀀스 처리 (embed_dim → hidden_size → output_size)
    - 마지막 타임스텝의 출력만 사용하여 감성 분류
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = CustomRNN(embed_dim, hidden_size, output_size, num_layers)

    def forward(self, x):
        # x: (batch, seq_len) 정수 인덱스
        embedded = self.embedding(x)         # (batch, seq_len, embed_dim)
        output, hidden = self.rnn(embedded)  # output: (batch, output_size) — 마지막 타임스텝만 반환됨
        return output                        # (batch, output_size)


# 옵티마이저, 손실함수
EMBED_DIM = 64
HIDDEN_SIZE = 128
OUTPUT_SIZE = 2       # 긍정(1) / 부정(0) 2클래스
NUM_LAYERS = 1

model = SentimentRNN(len(vocab), EMBED_DIM, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"\n모델 구조:\n{model}")


# 학습 및 평가
def train_model(model, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # 학습 모드
        total_loss = 0.0

        # Training Loop
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # 옵티마이저의 그라디언트 초기화
            outputs = model(inputs)  # 모델의 예측값 계산
            loss = criterion(outputs, labels)  # 손실 계산
            loss.backward()  # 역전파로 그라디언트 계산
            optimizer.step()  # 파라미터 업데이트

            total_loss += loss.item()

        # Epoch 별 평균 손실 출력
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

        # Validation Loop
        model.eval()  # 검증 모드
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # 검증에서는 그라디언트 계산이 필요 없음.
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Accuracy 계산
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Epoch 별 검증 손실 및 정확도 출력
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")


train_model(model, train_loader, val_loader, num_epochs=10)
