from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from .datatypes import Document, FeatureVector, Graph
from .graphing import build_adjacency, find_triangles, reduce_graph_to_triangles, build_graph

def _bitvector_from_triangles(n_nodes: int, triangles: List[Tuple[int,int,int]]) -> List[int]:
    mask = [0]*n_nodes
    for tri in triangles:
        for idx in tri:
            mask[idx] = 1
    return mask

def _pagerank_scores(graph: Graph, damping: float = 0.85, max_iter: int = 100, tolerance: float = 1e-6) -> List[float]:
    """
    Compute PageRank scores for sentences in the graph.
    
    PageRank Formula: PR(Si) = (1-d)/N + d × Σ(PR(Sj)/C(Sj))
    
    Args:
        graph: Graph with sentences as nodes and similarity edges
        damping: Damping factor (typically 0.85)
        max_iter: Maximum number of iterations
        tolerance: Convergence tolerance
    
    Returns:
        List of PageRank scores for each sentence
    """
    n = len(graph.nodes)
    if n == 0:
        return []
    
    # Initialize PageRank scores uniformly
    pr_scores = [1.0 / n] * n
    
    # Build adjacency list for efficient computation
    adj_list = [[] for _ in range(n)]
    out_degree = [0] * n
    
    for edge in graph.edges:
        # Add bidirectional edges (undirected graph)
        adj_list[edge.i].append(edge.j)
        adj_list[edge.j].append(edge.i)
        out_degree[edge.i] += 1
        out_degree[edge.j] += 1
    
    # Handle nodes with no outgoing edges (dangling nodes)
    for i in range(n):
        if out_degree[i] == 0:
            out_degree[i] = 1  # Avoid division by zero
    
    # PageRank iteration
    for iteration in range(max_iter):
        new_pr_scores = [(1.0 - damping) / n] * n
        
        for i in range(n):
            for j in adj_list[i]:
                # Add contribution from sentence j to sentence i
                new_pr_scores[i] += damping * (pr_scores[j] / out_degree[j])
        
        # Check for convergence
        diff = sum(abs(new_pr_scores[i] - pr_scores[i]) for i in range(n))
        pr_scores = new_pr_scores
        
        if diff < tolerance:
            break
    
    return pr_scores

def score_sentences(doc: Document, features: List[FeatureVector], graph=None, simM=None, threshold: float = 0.5) -> Tuple[List[float], List[int]]:
    # Ensure we have triangles and bitvector
    if graph is None and simM is not None:
        from .graphing import build_graph as _bg
        graph = _bg(doc, simM, threshold=threshold)
    assert graph is not None, "graph or simM must be provided"
    A = build_adjacency(graph)
    tris = find_triangles(A)
    bitvec = _bitvector_from_triangles(len(doc.sentences), tris)

    # Sum of selected feature scores (all six by default)
    scores: List[float] = []
    for i, f in enumerate(features):
        s = sum(f.values())
        scores.append(bitvec[i] * s)  # mask by bit-vector

    return scores, bitvec

def score_sentences_pagerank(doc: Document, 
                           features: List[FeatureVector], 
                           graph=None, 
                           simM=None, 
                           threshold: float = 0.5,
                           use_pagerank: bool = True,
                           pagerank_weight: float = 0.5,
                           damping: float = 0.85) -> Tuple[List[float], List[int]]:
    """
    Score sentences using a combination of features and PageRank.
    
    Args:
        doc: Document with sentences
        features: Feature vectors for each sentence
        graph: Pre-built graph (optional)
        simM: Similarity matrix (optional, used if graph is None)
        threshold: Similarity threshold for graph construction
        use_pagerank: Whether to include PageRank scores
        pagerank_weight: Weight for PageRank component (0.0 = only features, 1.0 = only PageRank)
        damping: PageRank damping factor
    
    Returns:
        Tuple of (scores, bitvector)
    """
    # Ensure we have graph
    if graph is None and simM is not None:
        from .graphing import build_graph as _bg
        graph = _bg(doc, simM, threshold=threshold)
    assert graph is not None, "graph or simM must be provided"
    
    # Get triangles and bitvector
    A = build_adjacency(graph)
    tris = find_triangles(A)
    bitvec = _bitvector_from_triangles(len(doc.sentences), tris)
    
    # Compute feature scores
    feature_scores = []
    for i, f in enumerate(features):
        s = sum(f.values())
        feature_scores.append(s)
    
    # Compute final scores
    if use_pagerank and len(graph.edges) > 0:
        # Compute PageRank scores
        pagerank_scores = _pagerank_scores(graph, damping=damping)
        
        # Normalize both score types to [0, 1] range
        if feature_scores:
            max_feature = max(feature_scores) if max(feature_scores) > 0 else 1.0
            feature_scores = [s / max_feature for s in feature_scores]
        
        if pagerank_scores:
            max_pagerank = max(pagerank_scores) if max(pagerank_scores) > 0 else 1.0
            pagerank_scores = [s / max_pagerank for s in pagerank_scores]
        
        # Combine feature scores and PageRank scores
        combined_scores = []
        for i in range(len(doc.sentences)):
            feature_component = (1.0 - pagerank_weight) * feature_scores[i]
            pagerank_component = pagerank_weight * pagerank_scores[i]
            combined_score = feature_component + pagerank_component
            combined_scores.append(bitvec[i] * combined_score)  # mask by bit-vector
    else:
        # Fall back to feature-only scoring
        combined_scores = [bitvec[i] * s for i, s in enumerate(feature_scores)]
    
    return combined_scores, bitvec
