# =============================================================================
# 독일어 → 영어 번역 모델 Fine-Tuning 전체 파이프라인
# 모델 : BERT
# 흐름: 데이터 로드 → 토크나이징 → Dataset/DataLoader 생성 → 학습 → 평가
# train_history = [2.8541, 2.3102, 1.9876, 1.7231, 1.5044, 1.3512, 1.2103, 1.1045, 1.0231, 0.9512]
# val_history   = [2.7123, 2.4501, 2.1893, 2.0102, 1.9234, 1.8901, 1.8756, 1.8820, 1.9012, 1.9345]
# 즉 각 에폭마다 LOSS 값이 저장되기 때문에 반환
# =============================================================================

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

#데이터 및 모델 로드 (허깅페이스에서 가져옴) 

# WMT14 독일어-영어 번역 데이터셋 로드 (학습/검증/테스트 세트 포함)
dataset = load_dataset("wmt14", "de-en")

# mBART: Meta가 만든 다국어 번역 모델 (50개 언어 지원)
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# mBART는 소스/타겟 언어 코드를 반드시 지정해야 함
# 테스크 목적은 번역
#   - 토크나이저가 입력 텍스트 앞에 언어 토큰을 자동 추가함
#   - 예: "de_DE" → 독일어 입력, "en_XX" → 영어 출력
tokenizer.src_lang = "de_DE"  # 소스 언어: 독일어
tokenizer.tgt_lang = "en_XX"  # 타겟 언어: 영어

# GPU가 있으면 GPU 사용, 없으면 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2단계: 커스텀 Dataset 클래스 정의
# 데이터로더 사용하려면 데이터셋 클래스가 있어야 함
# 역할은 원본 텍스트 -> 모델이 이해할 수 있는 토큰 ID 변환

class TranslationDataset(Dataset):
    """
    HuggingFace 데이터셋을 PyTorch Dataset으로 감싸는 클래스
    """

    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]["translation"]
        src_text = pair["de"]   # 입력 (독일어)
        tgt_text = pair["en"]   # 정답 (영어)

        # 입력 문장 토크나이징
        source = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 정답 문장 토크나이징
        target = self.tokenizer(
            text_target=tgt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 차원수 맞춰주는 중
        input_ids = source["input_ids"].squeeze(0)
        attention_mask = source["attention_mask"].squeeze(0)
        labels = target["input_ids"].squeeze(0)

        #패딩 CROSS ENTROPY LOSS 에서는 -100 으로 처리
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# 데이터로더 생성, 일부 데이터만 사용

TRAIN_SIZE = 5000  # 학습 데이터 수 (필요에 따라 조절)
VAL_SIZE = 500      # 검증 데이터 수
TEST_SIZE = 100     # 테스트 데이터 수
BATCH_SIZE = 8      # 한 번에 모델에 넣는 문장 수

# .select() : 특정 인덱스의 샘플만 골라서 부분 데이터셋을 만든다. dataset["train"].select(range(1000))
train_dataset = TranslationDataset(dataset["train"].select(range(TRAIN_SIZE)), tokenizer) # Dataset 클래스는 (데이터셋, 토크나이저) 형식
val_dataset = TranslationDataset(dataset["validation"].select(range(VAL_SIZE)), tokenizer)
test_dataset = TranslationDataset(dataset["test"].select(range(TEST_SIZE)), tokenizer)

# DataLoader: Dataset을 배치 단위로 잘라서 반복(iterate)할 수 있게 만들어줌
# shuffle=True: 매 에폭마다 데이터 순서를 섞어서 모델이 순서를 외우지 않게 함
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#옵티마이저 설정
# AdamW: Adam + Weight Decay (가중치 감쇠로 과적합 방지)
# lr=5e-5: 번역 모델 fine-tuning에 일반적으로 쓰이는 학습률
#
# criterion(손실함수)은 별도로 정의하지 않음!
# criterion = torch.nn.CrossEntropyLoss()
# mBART 같은 Seq2Seq 모델은 labels를 넘기면 내부에서
# CrossEntropyLoss를 자동 계산해서 outputs.loss로 반환함

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

#학습 루프

def train_model(num_epochs=10):
    """
    전체 학습 흐름 (에폭 반복)
    
    한 에폭의 흐름:
      [학습] 전체 train 데이터를 배치 단위로 순회하며 가중치 업데이트
      [검증] 전체 val 데이터로 현재 모델 성능 확인 (가중치 변경 없음)
    """
    train_history = []  # 에폭별 학습 손실 기록
    val_history = []    # 에폭별 검증 손실 기록

    for epoch in range(num_epochs):
        
        model.train()  # 학습 모드: Dropout, BatchNorm 등이 학습용으로 동작하게 만듦 학습할거다 지정
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"): #표시용
            # 각 올려야 할 것들을 디바이스에 올림
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 기울기 초기화
            optimizer.zero_grad()

            # 순전파 (Forward Pass)
            #   독일어 토큰 → 모델 통과 → 영어 예측 + 정답과 비교한 loss 자동 계산
            #   원본 코드의 model(inputs)와 달리, Seq2Seq 모델은
            #   input_ids, attention_mask, labels를 명시적으로 전달해야 함
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,  # labels를 넘기면 model이 loss를 자동 계산
            )

            # loss 추출
            #   원본 코드처럼 criterion(outputs, labels)를 쓸 필요 없음
            #    model이 이미 계산해서 outputs.loss에 담아줌
            loss = outputs.loss

            # 역전파 (Backward Pass)
            #  loss에서 출발해 각 파라미터의 기울기(gradient)를 계산
            #  "이 파라미터를 어느 방향으로 얼마나 바꿔야 loss가 줄어드는가"
            loss.backward()

            # 파라미터 업데이트
            #   계산된 기울기를 이용해 실제로 가중치를 수정
            optimizer.step()

            total_loss += loss.item()  # .item(): 텐서 → 파이썬 숫자 변환

        avg_train_loss = total_loss / len(train_loader)
        train_history.append(avg_train_loss)

        # 검증
        model.eval()  # 평가 모드: Dropout 비활성화, BatchNorm 고정
        val_loss = 0

        # 검증 시에는 학습 안시킬 거니까 기울기 계산 X
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_history.append(avg_val_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f}  "
            f"Val Loss: {avg_val_loss:.4f}"
        )

    return train_history, val_history


# 평가 루프

def evaluate_model():
    """
    학습이 끝난 뒤 테스트 데이터로 최종 성능 측정
    검증(validation)과 구조는 동일하지만, 한 번만 실행하는 최종 평가용
    """
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            test_loss += outputs.loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    return avg_test_loss


if __name__ == "__main__": #이 파일을 직접 실행했을 때만 실행
    print(f"Device: {device}") #CPU 냐 GPU 냐에 올린다. 
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}") #데이터 길이
    print("=" * 50)

    train_history, val_history = train_model(num_epochs=10)

    print("=" * 50)
    evaluate_model()