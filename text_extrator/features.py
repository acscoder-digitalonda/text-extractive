from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import math
from .datatypes import Document, FeatureVector

def _title_words(doc: Document) -> Set[str]:
    if doc.title and doc.title.strip():
        title_text = doc.title
    else:
        title_text = doc.sentences[0].text if doc.sentences else ""
    # reuse tokens from preprocessing if possible
    # here we lower & split simply:
    toks = [t.lower() for t in title_text.split()]
    return set(toks)

def _numerical_ratio(tokens: List[str]) -> float:
    nums = sum(1 for t in tokens if t.isdigit() or any(ch.isdigit() for ch in t))
    return nums / max(1, len(tokens))

def _sentence_length_norm(lengths: List[int]) -> List[float]:
    avg = sum(lengths) / max(1, len(lengths))
    return [l / max(1e-9, avg) for l in lengths]

def _position_scores(n: int) -> List[float]:
    # SP(Si) = (Total - position) / Total  ; position is 0-based idx
    return [(n - i) / n for i in range(n)]

def _thematic_words(doc: Document, top_k: int = 10) -> Set[str]:
    # simple TF across document (after preprocessing tokens were set)
    tf = Counter()
    for s in doc.sentences:
        tf.update(s.tokens)
    # select top_k tokens as thematic words
    return set([w for w, _ in tf.most_common(top_k)])

def _compute_tf(tokens: List[str], tf_mode: str = "sublinear") -> Dict[str, float]:
    """
    TF:
      - sublinear: 1 + log(count)
      - raw: count
      - norm: count / |d|
    """
    tf_counts = Counter(tokens)
    if not tf_counts:
        return {}

    if tf_mode == "sublinear":
        tf_scores = {t: (1.0 + math.log(c)) for t, c in tf_counts.items() if c > 0}
    elif tf_mode == "raw":
        tf_scores = dict(tf_counts)
    elif tf_mode == "norm":
        total = sum(tf_counts.values())
        tf_scores = {t: (c / total) for t, c in tf_counts.items()}
    else:
        raise ValueError(f"Unknown tf_mode: {tf_mode}")

    return tf_scores


def _compute_idf(doc, smooth_idf: bool = True) -> Dict[str, float]:
    """
    IDF:
      - smooth=True:  log((1+N)/(1+DF)) + 1  (giống scikit-learn)
      - smooth=False: log(N/DF)
    Ở đây mỗi câu được coi là 1 'document' để đo giống nhau giữa câu.
    """
    # gom toàn bộ term
    all_terms = set()
    for s in doc.sentences:
        all_terms.update(s.tokens)

    N = len(doc.sentences) if doc.sentences else 1
    idf: Dict[str, float] = {}

    for term in all_terms:
        DF = sum(1 for s in doc.sentences if term in s.tokens)
        if DF == 0:
            idf[term] = 0.0
            continue
        if smooth_idf:
            idf[term] = math.log((1.0 + N) / (1.0 + DF)) + 1.0
        else:
            # tránh chia 0; nếu DF==N thì idf=0 sẽ xảy ra (như định nghĩa gốc)
            idf[term] = math.log(N / DF) if DF > 0 else 0.0

    return idf


def _compute_tfidf_vector(tokens: List[str],
                          idf_scores: Dict[str, float],
                          tf_mode: str = "sublinear") -> Dict[str, float]:
    """TF-IDF(t,d) = TF(t,d) * IDF(t)"""
    tf_scores = _compute_tf(tokens, tf_mode=tf_mode)
    return {t: tf_scores[t] * idf_scores.get(t, 0.0) for t in tf_scores}


def _cosine_tfidf(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Cosine similarity cho vector thưa (dict term -> weight)"""
    if not v1 or not v2:
        return 0.0
    # giao nhau
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[t] * v2[t] for t in common)
    n1 = math.sqrt(sum(w*w for w in v1.values()))
    n2 = math.sqrt(sum(w*w for w in v2.values()))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)

def _cosine(v1: Counter, v2: Counter) -> float:
    # compute cosine over token counts (legacy function for backward compatibility)
    if not v1 or not v2:
        return 0.0
    # dot
    dot = sum(v1[t] * v2.get(t, 0) for t in v1)
    # norms
    n1 = sum(c*c for c in v1.values()) ** 0.5
    n2 = sum(c*c for c in v2.values()) ** 0.5
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)

def compute_similarity_matrix(doc,
                              use_tfidf: bool = True,
                              tf_mode: str = "sublinear",
                              smooth_idf: bool = True):
    """
    Trả về ma trận similarity giữa các câu.
    - use_tfidf=True: dùng TF-IDF + cosine
    - use_tfidf=False: cosine theo raw count (giữ nguyên hành vi cũ)
    Thêm tham số:
      - tf_mode: "sublinear" | "raw" | "norm"
      - smooth_idf: True | False
    """
    n = len(doc.sentences)
    if n == 0:
        return []

    if use_tfidf:
        # Use pre-computed TF-IDF vectors from sentences if available
        if all(hasattr(s, 'tf_idf_vector') and s.tf_idf_vector for s in doc.sentences):
            tfidf_vectors = [s.tf_idf_vector for s in doc.sentences]
        else:
            # Fallback: compute TF-IDF vectors on the fly
            idf_scores = _compute_idf(doc, smooth_idf=smooth_idf)
            tfidf_vectors = [
                _compute_tfidf_vector(s.tokens, idf_scores, tf_mode=tf_mode)
                for s in doc.sentences
            ]
        
        M = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                sim = _cosine_tfidf(tfidf_vectors[i], tfidf_vectors[j])
                M[i][j] = M[j][i] = sim
        return M
    else:
        # Legacy: cosine theo raw counts
        from collections import Counter
        vecs = [Counter(s.tokens) for s in doc.sentences]
        M = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                # dùng cosine theo count
                dot = sum(vecs[i][t]*vecs[j].get(t, 0) for t in vecs[i])
                n1 = math.sqrt(sum(c*c for c in vecs[i].values()))
                n2 = math.sqrt(sum(c*c for c in vecs[j].values()))
                sim = (dot / (n1*n2)) if n1 and n2 else 0.0
                M[i][j] = M[j][i] = sim
        return M

def extract_features(doc: Document, use_tfidf: bool = True) -> Tuple[List[FeatureVector], List[List[float]]]:
    n = len(doc.sentences)
    feats: List[FeatureVector] = [dict() for _ in range(n)]

    # Title words (simple overlap with title tokens, tokenized by spaces here)
    title_set = _title_words(doc)
    for i, s in enumerate(doc.sentences):
        overlap = sum(1 for tok in s.text.lower().split() if tok in title_set)
        feats[i]["title_words"] = overlap / max(1, len(title_set))

    # Sentence length (normalized by average length)
    lengths = [len(s.tokens) for s in doc.sentences]
    for i, val in enumerate(_sentence_length_norm(lengths)):
        feats[i]["sentence_length"] = val

    # Sentence position
    pos_scores = _position_scores(n)
    for i, val in enumerate(pos_scores):
        feats[i]["sentence_position"] = val

    # Numerical data ratio
    for i, s in enumerate(doc.sentences):
        feats[i]["numerical_data"] = _numerical_ratio(s.tokens)

    # Thematic words ratio
    thematic = _thematic_words(doc, top_k=max(5, n))  # heuristic
    max_tw = max(1, len(thematic))
    for i, s in enumerate(doc.sentences):
        count_tw = sum(1 for t in s.tokens if t in thematic)
        feats[i]["thematic_words"] = count_tw / max_tw

    # Sentence similarity (STS-like): sum of cosine sims normalized by max
    simM = compute_similarity_matrix(doc, use_tfidf=use_tfidf)
    for i in range(n):
        row = [simM[i][j] for j in range(n) if j != i]
        total = sum(row)
        m = max(row) if row else 1.0
        feats[i]["sentence_similarity"] = total / max(m, 1e-9)

    return feats, simM
