import numpy as np
from numpy.linalg import norm

# Load embeddings from a text file
embeddings = {}
file_path = "vectors.txt"

with open(file_path, "r") as f:
    first_line = f.readline()  # metadata line (ignored)
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        embeddings[word] = vector

print(f"Loaded {len(embeddings)} word vectors.")

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Print cosine similarity between king and queen
w1, w2 = 'king', 'queen'
sim = cosine_similarity(embeddings[w1], embeddings[w2])
print(f"Cosine similarity between '{w1}' and '{w2}': {sim:.4f}")

# Word analogy: king - man + woman = ?
def word_analogy(word_a, word_b, word_c, embeddings):
    """Find word closest to: embeddings[word_a] - embeddings[word_b] + embeddings[word_c]"""
    if word_a not in embeddings or word_b not in embeddings or word_c not in embeddings:
        return None

    target_vec = embeddings[word_a] - embeddings[word_b] + embeddings[word_c]

    best_word = None
    max_sim = -1
    for word, vec in embeddings.items():
        if word in [word_a, word_b, word_c]:
            continue
        sim = cosine_similarity(target_vec, vec)
        if sim > max_sim:
            max_sim = sim
            best_word = word
    return best_word

result = word_analogy('king', 'man', 'woman', embeddings)
print(f"'king' - 'man' + 'woman' = '{result}'")

# --- New part: print top 5 words similar to 'king' ---
def most_similar(word, embeddings, top_n=5):
    if word not in embeddings:
        return []
    similarities = []
    for w, vec in embeddings.items():
        if w == word:
            continue
        sim = cosine_similarity(embeddings[word], vec)
        similarities.append((w, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

similar_words = most_similar('king', embeddings, top_n=5)
print(f"Words most similar to 'king':")
for w, sim in similar_words:
    print(f"  {w} : {sim:.4f}")
