import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

class VGG16(torch.nn.Module):
    """
    VGG-16 모델 (MNIST 28x28 입력에 맞게 적응)
    MNIST 는 채널 1에 28*28
    원본 VGG-16: 13개 Conv층 + 3개 FC층 = 총 16개 가중치 층
    채널 구성: 64 → 128 → 256 → 512 → 512

    간단한 학습 결과

    [Epoch:    1] cost = 1.1071471
    [Epoch:    2] cost = 0.163525328
    [Epoch:    3] cost = 0.0956577361
    [Epoch:    4] cost = 0.0774243101
    [Epoch:    5] cost = 0.0846039504
    [Epoch:    6] cost = 0.0731596798
    [Epoch:    7] cost = 0.047487732
    [Epoch:    8] cost = 0.0639945418
    [Epoch:    9] cost = 0.0401985869
    [Epoch:   10] cost = 0.0445484146
    [Epoch:   11] cost = 0.0347271562
    """

    def __init__(self):
        super(VGG16, self).__init__()

        # Block 1: Conv(64) x2 + MaxPool
        # 입력: (batch, 1, 28, 28) → 출력: (batch, 64, 14, 14)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # Block 2: Conv(128) x2 + MaxPool
        # 입력: (batch, 64, 14, 14) → 출력: (batch, 128, 7, 7)
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # Block 3: Conv(256) x3 + MaxPool
        # 입력: (batch, 128, 7, 7) → 출력: (batch, 256, 3, 3)
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1), #각각 input channel, output channel 로 지정해준다. 
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # Block 4: Conv(512) x3 + MaxPool
        # 입력: (batch, 256, 3, 3) → 출력: (batch, 512, 1, 1)
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # Block 5: Conv(512) x3 (MaxPool 생략 — 이미 1x1이므로)
        # 입력: (batch, 512, 1, 1) → 출력: (batch, 512, 1, 1)
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU())

        # FC 층 (Classifier)
        # VGG-16 원본: 4096 → 4096 → 1000
        # MNIST 적응: 512 → 512 → 10
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 10)) #마지막 구분은 10개의 클래스 중 하나

    def forward(self, x):
        out = self.block1(x)       # Block 1 통과: 28x28 → 14x14
        out = self.block2(out)     # Block 2 통과: 14x14 → 7x7
        out = self.block3(out)     # Block 3 통과: 7x7 → 3x3
        out = self.block4(out)     # Block 4 통과: 3x3 → 1x1
        out = self.block5(out)     # Block 5 통과: 1x1 → 1x1
        out = out.view(out.size(0), -1)  # Flatten: (batch, 512)
        out = self.classifier(out) # FC 층 통과
        return out

# CNN 모델 정의
model = VGG16().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

# 학습을 진행하지 않을 것이므로 torch.no_grad() 왜냐하면 평가 단계라 학습된 모델로
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())