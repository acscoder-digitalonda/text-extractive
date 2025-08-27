from __future__ import annotations
from typing import List, Optional, Tuple
from .datatypes import Document
from .preprocessing import preprocess_text, PreprocessConfig
from .features import extract_features
from .graphing import build_graph, build_adjacency, find_triangles, reduce_graph_to_triangles
from .scoring import score_sentences

def generate_summary(doc: Document, scores: List[float], bitvec: List[int], compression_ratio: float = 0.2) -> str:
    n = len(doc.sentences)
     
    k = max(1, int(round(n * compression_ratio)))
    # pick indices with bitvec=1 and highest scores
    candidates = [(i, scores[i]) for i in range(n) if bitvec[i] == 1]
    # if not enough candidates (no triangles), fallback to top by scores regardless
    if len(candidates) < k:
        candidates = [(i, scores[i]) for i in range(n)]
    selected = [i for i,_ in sorted(candidates, key=lambda x: x[1], reverse=True)[:k]]
    selected.sort()  # restore original order
    return " ".join(doc.sentences[i].text for i in selected)

def summarize(text: str, title: Optional[str] = None, compression_ratio: float = 0.2, sim_threshold: float = 0.5) -> str:
    # Pipeline glue
    doc = preprocess_text(text, title=title, cfg=PreprocessConfig())
    feats, simM = extract_features(doc)
    graph = build_graph(doc, simM, threshold=sim_threshold)
    scores, bitvec = score_sentences(doc, feats, graph=graph, threshold=sim_threshold)

    summary = generate_summary(doc, scores, bitvec, compression_ratio=compression_ratio)
    return summary
