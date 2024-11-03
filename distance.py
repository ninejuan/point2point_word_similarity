from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# 유클리드 거리 계산 함수
def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2) ** 2))

# PCA를 통한 차원 축소 함수
def reduce_dimensions(embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)

# 임베딩 좌표를 2D 평면에 시각화하는 함수
def plot_embeddings(main_text, compare_text_arr, embeddings_2d):
    plt.figure(figsize=(10, 7))
    
    # 메인 단어 표시
    plt.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], color='blue', label=main_text, s=100)
    plt.text(embeddings_2d[0, 0], embeddings_2d[0, 1], main_text, color='blue', fontsize=12, ha='right')
    
    # 비교 대상 단어들 표시
    plt.scatter(embeddings_2d[1:, 0], embeddings_2d[1:, 1], color='orange', label="Comparison Words", s=100)
    for i, word in enumerate(compare_text_arr):
        plt.text(embeddings_2d[i + 1, 0], embeddings_2d[i + 1, 1], word, color='orange', fontsize=12, ha='right')

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.title("Embedding of Words in 2D Space")
    plt.grid(True)
    plt.show()

def visualize():
    embeddings = [get_embedding(text) for text in [main_text] + compare_text_arr]
    embedding_2d = reduce_dimensions(embeddings)
    print(embedding_2d[0])
    plot_embeddings(main_text, compare_text_arr, embedding_2d)

# 두 단어 입력
main_text = "Hapiness"
compare_text_arr = ["Sadness", "Anger", "Joy", "Fear", "Surprise", "Calmness", "Jealousy", "Satisfaction"]
main_embedding = get_embedding(main_text)

for word in compare_text_arr:
    word_embedding = get_embedding(word)
    distance = euclidean_distance(main_embedding, word_embedding)
    print(f"두 단어 '{main_text}'와 '{word}'의 거리 유사도: {distance}")

visualize()
# 2차원, 10차원, 768차원, 1536차원까지 모두 사용해서 비교해보기