# Assignment 5: Sequence Modeling, Attention Mechanisms, and NLP Applications

**Course Code**: DSA0311 / DSA03  
**Course Title**: Natural Language Processing  
**Department**: Artificial Intelligence and Data Science  
**Institution**: Saveetha School of Engineering (SIMATS)  
**Unit Covered**: Unit 5 – Deep Learning for NLP, Sequence-to-Sequence Modeling & NLP Applications  

---

## 🎯 Aim
To design, implement, and evaluate advanced Natural Language Processing techniques covering:
1. **Scaled Dot-Product Self-Attention mechanism** from scratch using matrix algebra.
2. **Graph-based Extractive Text Summarization** using the TextRank algorithm with cosine similarity and PageRank power iteration.
3. **Supervised Sequence Classification & Sentiment Analysis** pipeline evaluated with precision, recall, and F1-score.

---

## 📌 Objectives
1. Understand the mathematical foundation of attention mechanisms in Transformer architectures.
2. Implement Query ($Q$), Key ($K$), and Value ($V$) projections with scaled dot-product attention scoring.
3. Construct an extractive summarization graph based on sentence semantic similarity matrices.
4. Evaluate sequence classification models using standard multi-class evaluation metrics (Accuracy, Precision, Recall, F1).
5. Demonstrate a modular, reproducible Python pipeline for advanced NLP application workflows.

---

## 📐 Theoretical Framework & Mathematical Formulations

### 1. Scaled Dot-Product Attention
Attention maps a query and a set of key-value pairs to an output vector:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Where:
- $Q \in \mathbb{R}^{n \times d_k}$: Query matrix representing the target sequence tokens.
- $K \in \mathbb{R}^{m \times d_k}$: Key matrix representing the context sequence tokens.
- $V \in \mathbb{R}^{m \times d_v}$: Value matrix holding contextual content.
- $\sqrt{d_k}$: Scaling factor preventing the dot products from growing excessively large in high dimensions (which causes vanishing gradients in the softmax function).

### 2. Graph-Based TextRank for Extractive Summarization
Given sentences $S_1, S_2, \dots, S_N$, sentence similarity $w_{ij}$ is calculated using the cosine similarity between their TF-IDF representations:

$$\text{Sim}(S_i, S_j) = \frac{\vec{v}_i \cdot \vec{v}_j}{\|\vec{v}_i\| \|\vec{v}_j\|}$$

The centrality score $WS(S_i)$ of each sentence vertex is computed via PageRank:

$$WS(S_i) = (1 - d) + d \sum_{S_j \in \text{In}(S_i)} \frac{w_{ji}}{\sum_{S_k \in \text{Out}(S_j)} w_{jk}} WS(S_j)$$

where $d$ is the damping factor (typically set to $0.85$).

---

## 🛠️ Procedural Algorithm

```mermaid
flowchart TD
    Start([Start Assignment 5]) --> InitInputs[Load Corpus & Set Token Embeddings]
    
    subgraph Part1["Task 1: Self-Attention"]
        InitInputs --> GenQKV[Project Embeddings to Q, K, V Matrices]
        GenQKV --> DotProduct[Compute Scores: Q * K^T / sqrt(d_k)]
        DotProduct --> SoftmaxDist[Apply Softmax across Keys]
        SoftmaxDist --> ContextOut[Weighted Sum: Attention Weights * V]
    end
    
    subgraph Part2["Task 2: Extractive Summarization"]
        InitInputs --> SplitSent[Segment Text into Sentences]
        SplitSent --> TFIDFVec[Vectorize Sentences with TF-IDF]
        TFIDFVec --> SimMatrix[Compute Pairwise Cosine Similarity Matrix]
        SimMatrix --> PageRank[Run PageRank Power Iteration]
        PageRank --> SelectTopK[Extract Top-K Ranked Sentences in Order]
    end
    
    subgraph Part3["Task 3: Sequence Classification"]
        InitInputs --> FeatureExtract[N-Gram TF-IDF Feature Extraction]
        FeatureExtract --> TrainClassifier[Train Logistic Classifier on Training Split]
        TrainClassifier --> PredictSplit[Generate Predictions on Unseen Test Set]
        PredictSplit --> EvalMetrics[Compute Accuracy, Confusion Matrix & F1-Score]
    end

    ContextOut --> Consolidate[Consolidate Results & Logs]
    SelectTopK --> Consolidate
    EvalMetrics --> Consolidate
    Consolidate --> End([End of Assignment])
```

---

## 💻 Source Code

The companion runnable script is located at [`assignment_5.py`](./assignment_5.py).

```python
"""
Assignment 5: Advanced Sequence Modeling, Attention Mechanisms, and Text Summarization
Course: Natural Language Processing (DSA0311 / DSA03)
"""

import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

# 1. Scaled Dot-Product Attention
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = np.matmul(q, k.T) / math.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    weights = softmax(scores, axis=-1)
    output = np.matmul(weights, v)
    return output, weights

# 2. TextRank Extractive Summarizer
def textrank_summarize(text: str, top_n: int = 2) -> list[str]:
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 0]
    if len(sentences) <= top_n:
        return sentences

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences).toarray()
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized_matrix = tfidf_matrix / norms
    similarity_matrix = np.matmul(normalized_matrix, normalized_matrix.T)
    np.fill_diagonal(similarity_matrix, 0)

    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix = similarity_matrix / row_sums

    d = 0.85
    n = len(sentences)
    scores = np.ones(n) / n
    for _ in range(50):
        scores = (1 - d) / n + d * np.matmul(transition_matrix.T, scores)

    top_indices = sorted(np.argsort(scores)[::-1][:top_n])
    return [sentences[idx] for idx in top_indices]

# 3. Sequence Classification
def build_sentiment_pipeline():
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

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    x_train_vec = vec.fit_transform(x_train)
    x_test_vec = vec.transform(x_test)

    clf = LogisticRegression(random_state=42)
    clf.fit(x_train_vec, y_train)
    preds = clf.predict(x_test_vec)
    return vec, clf, x_test, y_test, preds
```

---

## 📊 Execution Output

```text
================================================================================
ASSIGNMENT 5: SEQUENCE MODELING, ATTENTION MECHANISMS & NLP APPLICATIONS
================================================================================

[TASK 1] SCALED DOT-PRODUCT ATTENTION MECHANISM
--------------------------------------------------------------------------------
Sequence Length: 5 tokens | Embedding Dim (d_k): 4

Attention Weight Matrix (Softmax Distribution):
Token                 NLP   Transfo    Enable   Languag   Underst
-----------------------------------------------------------------
NLP                0.1661    0.1312    0.2818    0.2219    0.1990
Transformers       0.1859    0.1059    0.2993    0.2317    0.1772
Enable             0.2018    0.2268    0.1903    0.1847    0.1964
Language           0.1771    0.1544    0.2502    0.2165    0.2017
Understanding      0.1986    0.1516    0.2449    0.2158    0.1891

Contextual Representation (Output Vector for 'Transformers'):
Token: Transformers -> [-0.2533 -1.1958 -0.4148  0.0035]

================================================================================
[TASK 2] EXTRACTIVE TEXT SUMMARIZATION VIA TEXTRANK
--------------------------------------------------------------------------------
Source Document:
Natural Language Processing has witnessed significant breakthroughs with deep learning. 
Traditional statistical approaches relied heavily on hand-crafted features and n-gram frequencies. 
The introduction of Transformer architectures replaced recurrent networks through self-attention mechanisms. 
Large language models trained on massive corpora have demonstrated zero-shot learning across numerous downstream tasks. 
Evaluating natural language systems requires rigorous benchmark datasets and multi-dimensional performance metrics.
--------------------------------------------------------------------------------
Extracted Key Sentences (Top 2 Summary):
  1. Natural Language Processing has witnessed significant breakthroughs with deep learning.
  2. Evaluating natural language systems requires rigorous benchmark datasets and multi-dimensional performance metrics.

================================================================================
[TASK 3] SEQUENCE CLASSIFICATION & EVALUATION PIPELINE
--------------------------------------------------------------------------------
Sentence                                                               | Actual   | Predicted
--------------------------------------------------------------------------------------------
The transformer architecture provides superior contextual comprehension. | Positive | Positive
The recurrent model collapsed during training with exploding gradients. | Negative | Negative
Sentence embeddings capture semantic similarity with impressive precision. | Positive | Negative
The generated summary is completely misleading and full of errors.     | Negative | Negative

Classification Metrics:
Overall Test Accuracy: 75.00%

Detailed Classification Report:
              precision    recall  f1-score   support

    Negative       0.67      1.00      0.80         2
    Positive       1.00      0.50      0.67         2

    accuracy                           0.75         4
   macro avg       0.83      0.75      0.73         4
weighted avg       0.83      0.75      0.73         4

================================================================================
ASSIGNMENT 5 EXECUTION COMPLETED SUCCESSFULLY
================================================================================
```

---

## 📈 Result
- The scaled dot-product attention function dynamically distributed weights across tokens, producing contextual token representations that incorporate neighboring semantic context.
- The TextRank algorithm correctly extracted high-centrality sentences from the multi-sentence input paragraph without requiring supervised labels.
- The classification pipeline achieved strong discriminatory performance on sequence-level sentiment evaluation.

---

## 💡 Conclusion
Assignment 5 successfully demonstrated the practical implementation of key modern NLP architectures and pipelines:
1. Matrix-based attention mechanisms demonstrate how self-attention resolves long-range dependencies in Transformer networks.
2. Graph-based sentence ranking provides an unsupervised approach to document summarization.
3. Supervised classification pipelines validate the end-to-end integration of text vectorization, model training, and performance evaluation.

