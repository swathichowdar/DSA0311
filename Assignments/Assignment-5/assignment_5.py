"""
Assignment 5: Advanced Sequence Modeling, Attention Mechanisms, and Text Summarization
Course: Natural Language Processing (DSA0311 / DSA03)
Department: Department of Artificial Intelligence and Data Science
Institution: Saveetha School of Engineering (SIMATS)
"""

import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# PART 1: SCALED DOT-PRODUCT ATTENTION (SELF-ATTENTION MECHANISM)
# ==============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Computes numerically stable softmax along the specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(
    q: np.ndarray, 
    k: np.ndarray, 
    v: np.ndarray, 
    mask: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    
    Parameters:
        q: Query matrix of shape (seq_len_q, d_k)
        k: Key matrix of shape (seq_len_k, d_k)
        v: Value matrix of shape (seq_len_v, d_v) where seq_len_k == seq_len_v
        mask: Optional mask matrix of shape (seq_len_q, seq_len_k)
        
    Returns:
        output: Contextual representations of shape (seq_len_q, d_v)
        attention_weights: Attention matrix of shape (seq_len_q, seq_len_k)
    """
    d_k = q.shape[-1]
    scores = np.matmul(q, k.T) / math.sqrt(d_k)
    
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
        
    attention_weights = softmax(scores, axis=-1)
    output = np.matmul(attention_weights, v)
    return output, attention_weights

# ==============================================================================
# PART 2: EXTRACTIVE TEXT SUMMARIZATION (GRAPH-BASED TEXTRANK ALGORITHM)
# ==============================================================================

def textrank_summarize(text: str, top_n: int = 2) -> list[str]:
    """
    Summarizes a multi-sentence text document using the Graph-based TextRank algorithm.
    1. Splits document into sentences.
    2. Builds TF-IDF representation for sentences.
    3. Computes sentence similarity matrix using cosine similarity.
    4. Computes PageRank scores iteratively.
    5. Returns top_n ranked sentences ordered chronologically.
    """
    import re
    # Sentence boundary split
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 0]
    if len(sentences) <= top_n:
        return sentences

    # Step 1: Compute TF-IDF matrix for sentences
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences).toarray()

    # Step 2: Compute Pairwise Cosine Similarity Adjacency Matrix
    # S_ij = dot(v_i, v_j) / (||v_i|| * ||v_j||)
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized_matrix = tfidf_matrix / norms
    similarity_matrix = np.matmul(normalized_matrix, normalized_matrix.T)

    # Set self-similarity diagonal to 0
    np.fill_diagonal(similarity_matrix, 0)

    # Row-normalize to transition probability matrix
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix = similarity_matrix / row_sums

    # Step 3: PageRank Power Iteration
    d = 0.85  # Damping factor
    n = len(sentences)
    scores = np.ones(n) / n
    for _ in range(50):
        scores = (1 - d) / n + d * np.matmul(transition_matrix.T, scores)

    # Step 4: Select top N sentences while preserving original order
    top_indices = np.argsort(scores)[::-1][:top_n]
    top_indices = sorted(top_indices)
    
    summary = [sentences[idx] for idx in top_indices]
    return summary

# ==============================================================================
# PART 3: SEQUENCE CLASSIFICATION PIPELINE (SENTIMENT ANALYSIS)
# ==============================================================================

def build_sentiment_pipeline():
    """
    Trains and validates a supervised sequence classification pipeline
    on domain-specific NLP task data.
    """
    training_data = [
        ("The attention mechanism revolutionized modern natural language processing.", 1),
        ("Transformer models achieve state-of-the-art results on NLP tasks.", 1),
        ("Extractive summarization successfully preserved the core meaning.", 1),
        ("Deep learning models demonstrate exceptional language understanding.", 1),
        ("The fine-tuning procedure improved classification accuracy significantly.", 1),
        ("Pretrained language models are remarkably versatile and effective.", 1),
        ("The model suffered from vanishing gradients and failed to converge.", 0),
        ("Out of vocabulary tokens resulted in inaccurate translations.", 0),
        ("The summarization output was fragmented and lacked grammatical coherence.", 0),
        ("Recurrent neural networks struggle with long-range sequence dependencies.", 0),
        ("The classification pipeline produced unacceptable prediction errors.", 0),
        ("High loss and overfitting degraded the model performance severely.", 0)
    ]

    test_data = [
        ("The transformer architecture provides superior contextual comprehension.", 1),
        ("The recurrent model collapsed during training with exploding gradients.", 0),
        ("Sentence embeddings capture semantic similarity with impressive precision.", 1),
        ("The generated summary is completely misleading and full of errors.", 0)
    ]

    x_train, y_train = zip(*training_data)
    x_test, y_test = zip(*test_data)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    classifier = LogisticRegression(random_state=42)
    classifier.fit(x_train_vec, y_train)
    predictions = classifier.predict(x_test_vec)

    return vectorizer, classifier, x_test, y_test, predictions

# ==============================================================================
# MAIN EXECUTION ROUTINE
# ==============================================================================

def main():
    print("=" * 80)
    print("ASSIGNMENT 5: SEQUENCE MODELING, ATTENTION MECHANISMS & NLP APPLICATIONS")
    print("=" * 80)

    # --- Task 1: Scaled Dot-Product Self-Attention ---
    print("\n[TASK 1] SCALED DOT-PRODUCT ATTENTION MECHANISM")
    print("-" * 80)
    
    tokens = ["NLP", "Transformers", "Enable", "Language", "Understanding"]
    seq_len = len(tokens)
    d_k = 4  # Embedding / Key dimension
    d_v = 4  # Value dimension
    
    np.random.seed(42)
    # Simulated token embeddings
    embeddings = np.array([
        [1.2, 0.4, 0.1, 0.8],  # NLP
        [0.9, 1.5, 0.7, 0.2],  # Transformers
        [0.2, 0.1, 1.4, 0.9],  # Enable
        [0.7, 0.8, 0.3, 1.1],  # Language
        [0.4, 1.2, 0.9, 0.6]   # Understanding
    ])
    
    # Weight matrices for Q, K, V projections
    W_Q = np.random.randn(d_k, d_k) * 0.5
    W_K = np.random.randn(d_k, d_k) * 0.5
    W_V = np.random.randn(d_k, d_v) * 0.5
    
    Q = np.matmul(embeddings, W_Q)
    K = np.matmul(embeddings, W_K)
    V = np.matmul(embeddings, W_V)
    
    context, attention_matrix = scaled_dot_product_attention(Q, K, V)
    
    print(f"Sequence Length: {seq_len} tokens | Embedding Dim (d_k): {d_k}")
    print("\nAttention Weight Matrix (Softmax Distribution):")
    header = f"{'Token':<15}" + "".join([f"{t[:7]:>10}" for t in tokens])
    print(header)
    print("-" * len(header))
    for i, token in enumerate(tokens):
        row_str = f"{token:<15}" + "".join([f"{attention_matrix[i, j]:10.4f}" for j in range(seq_len)])
        print(row_str)
        
    print("\nContextual Representation (Output Vector for 'Transformers'):")
    print(f"Token: {tokens[1]} -> {np.round(context[1], 4)}")

    # --- Task 2: Extractive Text Summarization ---
    print("\n" + "=" * 80)
    print("[TASK 2] EXTRACTIVE TEXT SUMMARIZATION VIA TEXTRANK")
    print("-" * 80)

    document = (
        "Natural Language Processing has witnessed significant breakthroughs with deep learning. "
        "Traditional statistical approaches relied heavily on hand-crafted features and n-gram frequencies. "
        "The introduction of Transformer architectures replaced recurrent networks through self-attention mechanisms. "
        "Large language models trained on massive corpora have demonstrated zero-shot learning across numerous downstream tasks. "
        "Evaluating natural language systems requires rigorous benchmark datasets and multi-dimensional performance metrics."
    )

    print("Source Document:")
    print(document)
    print("-" * 80)

    summary_sentences = textrank_summarize(document, top_n=2)
    print("Extracted Key Sentences (Top 2 Summary):")
    for idx, sentence in enumerate(summary_sentences, start=1):
        print(f"  {idx}. {sentence}")

    # --- Task 3: Sequence Classification & Evaluation ---
    print("\n" + "=" * 80)
    print("[TASK 3] SEQUENCE CLASSIFICATION & EVALUATION PIPELINE")
    print("-" * 80)

    vectorizer, model, x_test, y_test, preds = build_sentiment_pipeline()

    print(f"{'Sentence':<70} | {'Actual':<8} | {'Predicted':<8}")
    print("-" * 92)
    for sent, actual, pred in zip(x_test, y_test, preds):
        act_label = "Positive" if actual == 1 else "Negative"
        pred_label = "Positive" if pred == 1 else "Negative"
        print(f"{sent:<70} | {act_label:<8} | {pred_label:<8}")

    acc = accuracy_score(y_test, preds)
    print("\nClassification Metrics:")
    print(f"Overall Test Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, preds, target_names=["Negative", "Positive"]))

    print("=" * 80)
    print("ASSIGNMENT 5 EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()

