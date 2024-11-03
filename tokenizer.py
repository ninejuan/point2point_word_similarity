from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# 모델 및 토크나이저 설정
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 텍스트를 임베딩 벡터로 변환하는 함수
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=2)

main_text = "Hapiness"
compare_text_arr = ["Sadness", "Anger", "Joy", "Fear", "Surprise", "Calmness", "Jealousy", "Satisfaction"]
main_embedding = get_embedding(main_text)
print(main_embedding)
