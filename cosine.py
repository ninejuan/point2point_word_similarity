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
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def cosine_sim(vector1, vector2):
    # Convert lists to numpy arrays
    vec1 = np.array(vector1)
    vec2 = np.array(vector2)

    # Calculate the dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate the magnitudes (norms) of each vector
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # Calculate cosine similarity
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # Return 0 if one of the vectors is zero to avoid division by zero

    return dot_product / (norm_vec1 * norm_vec2)


main_text = "Hapiness"
compare_text_arr = ["Sadness", "Anger", "Joy", "Fear", "Surprise", "Calmness", "Jealousy", "Satisfaction"]
main_embedding = get_embedding(main_text)

for word in compare_text_arr:
    word_embedding = get_embedding(word)
    distance = cosine_sim(main_embedding, word_embedding)
    print(f"두 단어 '{main_text}'와 '{word}'의 코사인 유사도: {distance}")
