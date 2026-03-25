# 1. 데이터 로드 및 단어 토큰화

import urllib.request
import numpy as np
from tqdm import tqdm
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#데이터 불러오기

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20RNN%20Sequence%20Labeling/dataset/train.txt", filename="train.txt")

f = open('train.txt', 'r')
tagged_sentences = []
sentence = []

# 파일 읽고 전처리 
'''
- 문장 경계 시작 이후에 있을때 sentence 버퍼가 쌓여 있으면 쌓인거를 -> tagged_sentence 에 옮기고 바로 비움
- 공백 기준으로 단어 나누고 줄바꿈 제거하고 전체 코퍼스 중 단어와 개체명 태깅만 기록
'''
for line in f:
    if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n": #문장의 경계임을 지정
        if len(sentence) > 0: #sentence 버퍼가 쌓여 있으면 즉 ["eu", "B-ORG"] 이런 형태로 저장된게 있다면
            tagged_sentences.append(sentence) #tagged_sentences 에 옮겨 추가
            sentence = [] #바로 비움
        continue
    splits = line.split(' ') # 공백을 기준으로 속성
    splits[-1] = re.sub(r'\n', '', splits[-1]) # 줄바꿈 표시 \n 제거, (찾을것, 바꿀것, 대상 문자열)
    word = splits[0].lower() # 단어는 소문자
    sentence.append([word, splits[-1]]) # 단어와 개체명 태깅만 기록

print("전체 샘플 개수: ", len(tagged_sentences)) # 전체 샘플의 개수 출력

print(tagged_sentences[0]) # 첫번째 샘플 출력

# train, test, val 분리 준비
#scikit-learn 에 train_test_split 함수 존재
# train_test_split(뭘 나눌지, 뭘 나눌지, test_size, random_state 로 고정)
sentences, ner_tags = [], []
for tagged_sentence in tagged_sentences: # 14,041개의 문장 샘플을 1개씩 불러온다.
    sentence, tag_info = zip(*tagged_sentence) # 각 샘플에서 단어들은 sentence에 개체명 태깅 정보들은 
    #tag_info에 저장.
    sentences.append(list(sentence)) # 각 샘플에서 단어 정보만 저장 -> X 지정
    ner_tags.append(list(tag_info)) # 각 샘플에서 개체명 태깅 정보만 저장 -> y 지정



X_train, X_test, y_train, y_test = train_test_split(sentences, ner_tags, test_size=.2, random_state=777)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=.2, random_state=777)

print('훈련 데이터의 개수 :', len(X_train))
print('검증 데이터의 개수 :', len(X_valid))
print('테스트 데이터의 개수 :', len(X_test))
print('훈련 데이터 레이블의 개수 :', len(X_train))
print('검증 데이터 레이블의 개수 :', len(X_valid))
print('테스트 데이터 레이블의 개수 :', len(X_test))


# Vocab 만들기 (단어 사전 구축)

word_list = []
for sent in X_train:
    for word in sent:
      word_list.append(word)

#빈도 기반으로 
'''
Counter({'나는': 3, '학생': 2, '공부': 1}) 이렇게 저장됨
'''

word_counts = Counter(word_list)
print('총 단어수 :', len(word_counts))

print('훈련 데이터에서의 단어 the의 등장 횟수 :', word_counts['the'])
print('훈련 데이터에서의 단어 love의 등장 횟수 :', word_counts['love'])

vocab = sorted(word_counts, key=word_counts.get, reverse=True)
print('등장 빈도수 상위 10개 단어')
print(vocab[:10])

# 단어 -> 인덱스.
'''
이때 패딩도 인덱스 부여 과정
<PAD> = 0
<UNK> = 1
'''
word_to_index = {}
word_to_index['<PAD>'] = 0
word_to_index['<UNK>'] = 1

for index, word in enumerate(vocab) :
  word_to_index[word] = index + 2

vocab_size = len(word_to_index)
print('패딩 토큰과 UNK 토큰을 고려한 단어 집합의 크기 :', vocab_size)

print('단어 <PAD>와 맵핑되는 정수 :', word_to_index['<PAD>'])
print('단어 <UNK>와 맵핑되는 정수 :', word_to_index['<UNK>'])
print('단어 the와 맵핑되는 정수 :', word_to_index['the'])

print(word_to_index) # 단어 사전 구축 완료


# 정수를 인덱스로 변환 함수 지정

def texts_to_sequences(tokenized_X_data, word_to_index):
  encoded_X_data = []
  for sent in tokenized_X_data: # 단어 하나씩 추출
    index_sequences = []
    for word in sent:
      try:
          index_sequences.append(word_to_index[word])
      except KeyError:
          index_sequences.append(word_to_index['<UNK>']) #사전에 없으면 <UNK> 인덱스로 부여
    encoded_X_data.append(index_sequences)
  return encoded_X_data

#train, test, val 모두 인토딩 함수 적용해서 인덱스화 완료
encoded_X_train = texts_to_sequences(X_train, word_to_index)
encoded_X_valid = texts_to_sequences(X_valid, word_to_index)
encoded_X_test = texts_to_sequences(X_test, word_to_index)

# 상위 샘플 2개 출력
for sent in encoded_X_train[:2]:
  print(sent)

# y데이터는 아직 정수 인코딩이 되지 않은 상태
for sent in y_train[:2]:
  print(sent)

index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key

print(index_to_word)

decoded_sample = [index_to_word[word] for word in encoded_X_train[0]]
print('기존의 첫번째 샘플 :', X_train[0])
print('복원된 첫번째 샘플 :', decoded_sample)

# y_train으로부터 존재하는 모든 태그들의 집합 구하기
flatten_tags = [tag for sent in y_train for tag in sent]
tag_vocab = list(set(flatten_tags))
print('태그 집합 :', tag_vocab)
print('태그 집합의 크기 :', len(tag_vocab))

tag_to_index = {}
tag_to_index['<PAD>'] = 0

for index, word in enumerate(tag_vocab) :
  tag_to_index[word] = index + 1

tag_vocab_size = len(tag_to_index)
# print('패딩 토큰까지 포함된 태그 집합의 크기 :', tag_vocab_size)
print('태그 집합 :', tag_to_index)

def encoding_label(sequence, tag_to_index):
  label_sequence = []
  for seq in sequence:
    label_sequence.append([tag_to_index[tag] for tag in seq])
  return label_sequence

# 출력 데이터도 인코딩
encoded_y_train = encoding_label(y_train, tag_to_index)
encoded_y_valid = encoding_label(y_valid, tag_to_index)
encoded_y_test = encoding_label(y_test, tag_to_index)

#패딩 지정

print('샘플의 최대 길이 : %d' % max(len(l) for l in encoded_X_train))
print('샘플의 평균 길이 : %f' % (sum(map(len, encoded_X_train))/len(encoded_X_train)))

#최대 길이 출력 했을때 최대 길이보다 약간 더 길게 max_len 지정하고 차이 부분은 0 패딩 적용

max_len = 115 #공식 사이트에 올라간 버전은 가장 긴 문장이 78 단어였으나 현재 버전은 113? 인가 여서 115 를 max_len 지정

def pad_sequences(sentences, max_len):
    features = np.zeros((len(sentences), max_len), dtype=int) #0 초기화 -> 패딩 적용 준비
    for index, sentence in enumerate(sentences):
        if len(sentence) != 0: #문장이 비어있지 않다면
            features[index, :len(sentence)] = np.array(sentence)[:max_len] 
            '''
            문장이 max_len 보다 짧으면 나머지 남은 부분만큼 np.zeros 의 0 으로 둠
            features[index, a] 는 인덱스 수만큼 문장 있고, a 만큼 단어 있는거
            features = [[3,  5,  8,  0],   ← index=0 (행)
             [12, 7,  0,  0],   ← index=1 (행)
             [1,  4,  6,  9]]   ← index=2 (행)
              ↑   ↑   ↑   ↑
             a=0 a=1 a=2 a=3 (열)
            '''
    return features # 패딩 적용 완료

# 전체 패딩 적용
padded_X_train = pad_sequences(encoded_X_train, max_len=max_len)
padded_X_valid = pad_sequences(encoded_X_valid, max_len=max_len)
padded_X_test = pad_sequences(encoded_X_test, max_len=max_len)

padded_y_train = pad_sequences(encoded_y_train, max_len=max_len)
padded_y_valid = pad_sequences(encoded_y_valid, max_len=max_len)
padded_y_test = pad_sequences(encoded_y_test, max_len=max_len)


# 모델링

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", device)

#LSTM 모델 정의
class NERTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2):
        super(NERTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_length, hidden_dim*2)
        logits = self.fc(lstm_out)  # (batch_size, seq_length, output_dim)
        return logits

# 데이터 텐서 변환
X_train_tensor = torch.tensor(padded_X_train, dtype=torch.long)
y_train_tensor = torch.tensor(padded_y_train, dtype=torch.long)
X_valid_tensor = torch.tensor(padded_X_valid, dtype=torch.long)
y_valid_tensor = torch.tensor(padded_y_valid, dtype=torch.long)
X_test_tensor = torch.tensor(padded_X_test, dtype=torch.long)
y_test_tensor = torch.tensor(padded_y_test, dtype=torch.long)

# Dataset 클래스애 올리고 DataLoader 클래스에 넣어서 배치 단위로 잘라서 반복(iterate)할 수 있게 만들어줌

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=32)
valid_dataset = torch.utils.data.TensorDataset(X_valid_tensor, y_valid_tensor)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, batch_size=32)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=32)

print(vocab_size)

# 하이퍼파리미터 정의

embedding_dim = 100
hidden_dim = 256
output_dim = tag_vocab_size
learning_rate = 0.01
num_epochs = 10
num_layers = 2

# Model, loss, optimizer
model = NERTagger(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 원래 내장 함수로 accuracy 불러와서 찍을 수 있는데 패딩 토큰 때문에 오류 발생 가능 -> 직접 구현
def calculate_accuracy(logits, labels, ignore_index=0):
    # 예측 레이블을 구합니다.
    predicted = torch.argmax(logits, dim=1)

    # 패딩 토큰은 무시합니다.
    mask = (labels != ignore_index)

    # 정답을 맞춘 경우를 집계합니다.
    correct = (predicted == labels).masked_select(mask).sum().item()
    total = mask.sum().item()

    accuracy = correct / total
    return accuracy

# 평가 루프, val 데이터에 대해

def evaluate(model, valid_dataloader, criterion, device):
    val_loss = 0
    val_correct = 0
    val_total = 0

    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in valid_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            logits = model(batch_X)

            # Compute loss
            loss = criterion(logits.view(-1, output_dim), batch_y.view(-1))

            # Calculate validation accuracy and loss
            val_loss += loss.item()
            val_correct += calculate_accuracy(logits.view(-1, output_dim), batch_y.view(-1)) * batch_y.size(0)
            val_total += batch_y.size(0)

    val_accuracy = val_correct / val_total
    val_loss /= len(valid_dataloader)

    return val_loss, val_accuracy

# Train loop

best_val_loss = float('inf')
'''
val 은 학습을 하지 않음
model.train() 이든 loss 계산이든 optimizer 적용이든 다 X -> val 전에 model.eval() 으로 변경

val로 하는 구체적인 판단들
Early stopping: val_loss 올라가면 학습 중단
모델 저장: val_loss가 가장 낮았던 시점의 가중치를 저장
하이퍼파라미터 튜닝: learning rate, 모델 크기 등 조정

<자동 하이퍼파라미터 튜닝 방법> 

Grid Search가 가장 단순
후보 값을 미리 정해두고 모든 조합을 다 시도

예를 들어 learning_rate=[0.01, 0.001], hidden_dim=[128, 256]이면 
4가지 조합을 전부 돌려보고 val 성능이 가장 좋은 걸 고르는 구조.
확실하지만 조합이 많아지면 시간이 폭발적으로 증가

Random Search는 조합을 무작위로 샘플링해서 시도
의외로 Grid Search보다 효율이 좋은 경우가 많은데, 
실제로 중요한 하이퍼파라미터는 몇 개 안 되기 때문에 랜덤으로 넓게 탐색하는 게 유리

Bayesian Optimization은 가장 똑똑한 방식입니다. 
이전 시도 결과를 바탕으로 "다음에 어떤 값을 시도하면 좋을지" 확률적으로 예측해서 탐색
무작정 다 해보는 게 아니라 유망한 영역을 집중 탐색하니까 적은 시도로 좋은 결과 얻을 수 있음

실제로 많이 쓰는 라이브러리는 Optuna
'''

for epoch in range(num_epochs):
    # Training
    train_loss = 0
    train_correct = 0
    train_total = 0
    model.train()
    for batch_X, batch_y in train_dataloader:
        # Forward pass
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        logits = model(batch_X)

        # Compute loss
        loss = criterion(logits.view(-1, output_dim), batch_y.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy and loss
        train_loss += loss.item()
        train_correct += calculate_accuracy(logits.view(-1, output_dim), batch_y.view(-1)) * batch_y.size(0)
        train_total += batch_y.size(0)

    train_accuracy = train_correct / train_total
    train_loss /= len(train_dataloader)

    # Validation
    val_loss, val_accuracy = evaluate(model, valid_dataloader, criterion, device)

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # 검증 손실이 최소일 때 체크포인트 저장
    if val_loss < best_val_loss:
        print(f'Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. 체크포인트를 저장합니다.')
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_checkpoint.pth') #별도 위치 지정 안했으면 스크립트 실행 위치에 저장
        # 최고 성능일때 모델의 모든 가중치 딕셔너리로 저장

# 모델 로드 적용 및 평가

# 모델 로드
model.load_state_dict(torch.load('best_model_checkpoint.pth'))

# 모델을 device에 올립니다.
model.to(device)

# 검증 데이터에 대한 정확도와 손실 계산
val_loss, val_accuracy = evaluate(model, valid_dataloader, criterion, device)

print(f'Best model validation loss: {val_loss:.4f}')
print(f'Best model validation accuracy: {val_accuracy:.4f}')

# 테스트 데이터에 대한 정확도와 손실 계산
test_loss, test_accuracy = evaluate(model, test_dataloader, criterion, device)

print(f'Best model test loss: {test_loss:.4f}')
print(f'Best model test accuracy: {test_accuracy:.4f}')

# 테스팅

index_to_tag = {}
for key, value in tag_to_index.items():
    index_to_tag[value] = key

def predict_labels(text, model, word_to_ix, index_to_tag, max_len=150):
    # 단어 토큰화
    tokens = text.split()

    # 정수 인코딩
    token_indices = [word_to_ix.get(token, 1) for token in tokens]

    # 패딩
    token_indices_padded = np.zeros(max_len, dtype=int)
    token_indices_padded[:len(token_indices)] = token_indices[:max_len]

    # 텐서로 변환
    input_tensor = torch.tensor(token_indices_padded, dtype=torch.long).unsqueeze(0).to(device)

    # 모델의 입력으로 사용하고 예측값 리턴
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)

    # 가장 값이 높은 인덱스를 예측값으로 선택
    predicted_indices = torch.argmax(logits, dim=-1).squeeze(0).tolist()

    # 패딩 토큰 제거
    predicted_indices_no_pad = predicted_indices[:len(tokens)]

    # 패딩 토큰을 제외하고 정수 시퀀스를 예측 시퀀스로 변환
    predicted_tags = [index_to_tag[index] for index in predicted_indices_no_pad]

    return predicted_tags

print(X_test[0])

sample = ' '.join(X_test[0])
print(sample)

predicted_tags = predict_labels(sample, model, word_to_index, index_to_tag) # 테스트 실행
