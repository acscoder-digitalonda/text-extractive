from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

@dataclass
class Sentence:
    idx: int
    text: str
    tokens: List[str] = field(default_factory=list)
    tf_idf_vector: Dict[str, float] = field(default_factory=dict)

@dataclass
class Document:
    title: Optional[str]
    raw_text: str
    sentences: List[Sentence]

@dataclass
class Edge:
    i: int
    j: int
    weight: float  # similarity

@dataclass
class Graph:
    nodes: List[Sentence]
    edges: List[Edge]  # undirected weighted edges

FeatureVector = Dict[str, float]  # per-sentence feature scores
