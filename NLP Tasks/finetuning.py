# 독일어 - 영어 쌍 데이터를 허깅페이스에서 모델 가지고 와서 fine-tuning 학습시키고 테스트까지 해보기
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 데이터셋 및 모델 로드

dataset = load_dataset("wmt14", "de-en")
model_name = "facebook/mbart-large-50-many-to-many-mmt"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)