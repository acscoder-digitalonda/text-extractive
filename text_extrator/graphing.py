from __future__ import annotations
from typing import List, Tuple, Set
from .datatypes import Document, Graph, Edge, Sentence

def build_graph(doc: Document, simM: List[List[float]], threshold: float = 0.5) -> Graph:
    nodes = doc.sentences
    edges: List[Edge] = []
    n = len(nodes)
    for i in range(n):
        for j in range(i+1, n):
            w = simM[i][j]
            if w >= threshold:
                edges.append(Edge(i=i, j=j, weight=w))
    return Graph(nodes=nodes, edges=edges)

def build_adjacency(graph: Graph) -> List[List[int]]:
    n = len(graph.nodes)
    A = [[0]*n for _ in range(n)]
    for e in graph.edges:
        A[e.i][e.j] = 1
        A[e.j][e.i] = 1
    return A

def find_triangles(A: List[List[int]]) -> List[Tuple[int,int,int]]:
    n = len(A)
    tris: List[Tuple[int,int,int]] = []
    for x in range(n):
        for y in range(x+1, n):
            if A[x][y] == 0: 
                continue
            for z in range(y+1, n):
                if A[y][z] and A[x][z]:
                    tris.append((x,y,z))
    return tris

def reduce_graph_to_triangles(graph: Graph, triangles: List[Tuple[int,int,int]]) -> Graph:
    keep_edges: Set[Tuple[int,int]] = set()
    for x,y,z in triangles:
        keep_edges.update({(min(x,y), max(x,y)), (min(y,z), max(y,z)), (min(x,z), max(x,z))})
    edges = []
    for e in graph.edges:
        key = (min(e.i, e.j), max(e.i, e.j))
        if key in keep_edges:
            edges.append(e)
    return Graph(nodes=graph.nodes, edges=edges)
