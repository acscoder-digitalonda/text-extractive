from __future__ import annotations
import streamlit as st
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
import io

from text_extrator.summarize import summarize, generate_summary
from text_extrator.preprocessing import preprocess_text, PreprocessConfig
from text_extrator.features import extract_features
from text_extrator.graphing import build_graph, build_adjacency, find_triangles, reduce_graph_to_triangles
from text_extrator.scoring import score_sentences

def extract_rtf_text(rtf_content):
    """Extract plain text from RTF content."""
    # Remove RTF control words and groups
    text = re.sub(r'\\[a-z]+\d*', '', rtf_content)
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r'\\\*.*?;', '', text)
    text = re.sub(r'\\[^a-z]', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_markdown_text(md_content):
    """Extract plain text from Markdown content."""
    # Remove headers
    text = re.sub(r'^#{1,6}\s+', '', md_content, flags=re.MULTILINE)
    # Remove bold and italic
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
    text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)
    # Remove links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def generate_default_title(filename: str) -> str:
    """Generate default title from filename."""
    # Remove file extension
    name = filename.rsplit('.', 1)[0]
    # Replace underscores and hyphens with spaces
    name = name.replace('_', ' ').replace('-', ' ')
    # Capitalize first letter of each word
    return ' '.join(word.capitalize() for word in name.split())

def load_text_from_file(uploaded_file):
    """Load text content from uploaded file based on file type."""
    file_extension = uploaded_file.name.lower().split('.')[-1]
    content = uploaded_file.read().decode("utf-8")
    
    if file_extension == 'rtf':
        return extract_rtf_text(content)
    elif file_extension == 'md':
        return extract_markdown_text(content)
    else:  # txt and other formats
        return content

def draw_graph_visualization(graph, triangles, doc, simM, sim_threshold):
    """Create graph visualization showing nodes, edges, and triangles."""
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes (sentences)
    for i, sentence in enumerate(doc.sentences):
        # Truncate sentence text for display
        label = f"S{i+1}"
        preview = sentence.text[:30] + "..." if len(sentence.text) > 30 else sentence.text
        G.add_node(i, label=label, preview=preview)
    
    # Add edges with weights
    for edge in graph.edges:
        G.add_edge(edge.i, edge.j, weight=edge.weight)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Full graph with all edges
    ax1.set_title("Full Graph (All Edges Above Threshold)", fontsize=14, fontweight='bold')
    
    if len(G.nodes) > 0:
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax1, 
                              node_color='lightblue', 
                              node_size=800,
                              alpha=0.7)
        
        # Draw edges with thickness based on weight
        edges = G.edges(data=True)
        if edges:
            weights = [edge[2]['weight'] for edge in edges]
            max_weight = max(weights) if weights else 1
            edge_widths = [3 * (w / max_weight) for w in weights]
            
            nx.draw_networkx_edges(G, pos, ax=ax1,
                                  width=edge_widths,
                                  alpha=0.6,
                                  edge_color='gray')
        
        # Draw labels
        labels = {i: f"S{i+1}" for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=10, font_weight='bold')
        
        # Add edge weight labels for small graphs
        if len(G.nodes) <= 10:
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1, font_size=8)
    
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Plot 2: Triangles highlighted
    ax2.set_title("Triangles Highlighted", fontsize=14, fontweight='bold')
    
    if len(G.nodes) > 0:
        # Draw the same graph
        nx.draw_networkx_nodes(G, pos, ax=ax2,
                              node_color='lightblue',
                              node_size=800,
                              alpha=0.7)
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(G, pos, ax=ax2,
                              width=1,
                              alpha=0.3,
                              edge_color='lightgray')
        
        # Highlight triangle edges in different colors
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
        for tri_idx, (x, y, z) in enumerate(triangles[:8]):  # Limit to 8 triangles for color variety
            color = colors[tri_idx % len(colors)]
            
            # Draw triangle edges
            triangle_edges = [(x, y), (y, z), (x, z)]
            nx.draw_networkx_edges(G, pos, triangle_edges, ax=ax2,
                                  width=3,
                                  alpha=0.8,
                                  edge_color=color)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels, ax=ax2, font_size=10, font_weight='bold')
        
        # Highlight triangle nodes
        triangle_nodes = set()
        for tri in triangles:
            triangle_nodes.update(tri)
        
        if triangle_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=list(triangle_nodes), ax=ax2,
                                  node_color='yellow',
                                  node_size=1000,
                                  alpha=0.8)
    
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Add legend for triangles
    if triangles:
        legend_elements = []
        for tri_idx, (x, y, z) in enumerate(triangles[:8]):
            color = colors[tri_idx % len(colors)]
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=3, 
                                            label=f'Triangle {tri_idx+1}: S{x+1}-S{y+1}-S{z+1}'))
        
        if len(triangles) > 8:
            legend_elements.append(plt.Line2D([0], [0], color='black', lw=1, 
                                            label=f'... and {len(triangles)-8} more'))
        
        ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    # Convert plot to image for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def create_sidebar_controls():
    """Create sidebar controls for parameters."""
    st.sidebar.header("Parameters")
    ratio = st.sidebar.slider(
        "Compression ratio", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.2, 
        step=0.1,
        help="Compression ratio (0-1)"
    )
    threshold = st.sidebar.slider(
        "Similarity threshold", 
        min_value=0.05, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Similarity threshold for edges"
    )
    
    st.sidebar.header("Debug Options")
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=True, help="Show detailed pipeline steps")
    
    return ratio, threshold, debug_mode

def debug_pipeline(text: str, title: str, compression_ratio: float, sim_threshold: float):
    """Run the pipeline with detailed debugging information."""
    
    # Step 1: Pre-processing
    st.header("🔧 Step 1: Pre-processing")
    with st.expander("Pre-processing Details", expanded=True):
        st.write("**Running:** Tokenization, Normalization, Stop-word removal, Stemming, TF-IDF computation")
        
        with st.spinner("Processing text..."):
            doc = preprocess_text(text, title=title, cfg=PreprocessConfig())
            
            # Get TF-IDF vectors from sentences (now stored during preprocessing)
            tfidf_vectors = [s.tf_idf_vector for s in doc.sentences]
            
            # Get IDF scores for debugging display
            from text_extrator.features import _compute_idf
            idf_scores = _compute_idf(doc, smooth_idf=True)
        
        st.success(f"✅ Processed {len(doc.sentences)} sentences")
        
        # Show preprocessing results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sentences", len(doc.sentences))
            st.metric("Total Words (original)", len(text.split()))
        with col2:
            total_tokens = sum(len(s.tokens) for s in doc.sentences)
            st.metric("Total Tokens (processed)", total_tokens)
            st.metric("Stopwords Removed", len(text.split()) - total_tokens)
        with col3:
            st.metric("Unique Terms", len(idf_scores))
            avg_tfidf_terms = sum(len(vec) for vec in tfidf_vectors) / len(tfidf_vectors) if tfidf_vectors else 0
            st.metric("Avg TF-IDF Terms/Sentence", f"{avg_tfidf_terms:.1f}")
        
        # Show sentence breakdown with TF-IDF info
        sentences_data = []
        for i, s in enumerate(doc.sentences):
            tfidf_vec = tfidf_vectors[i]
            # Get top 5 TF-IDF terms for this sentence
            top_tfidf = sorted(tfidf_vec.items(), key=lambda x: x[1], reverse=True)[:5]
            tfidf_display = ", ".join([f"{term}:{score:.3f}" for term, score in top_tfidf])
            
            sentences_data.append({
                "Sentence #": i+1,
                "Original Text": s.text[:80] + "..." if len(s.text) > 80 else s.text,
                "Tokens": len(s.tokens),
                "Processed Tokens": ", ".join(s.tokens[:8]) + ("..." if len(s.tokens) > 8 else ""),
                "Top TF-IDF Terms": tfidf_display if tfidf_display else "No terms"
            })
        
        sentences_df = pd.DataFrame(sentences_data)
        st.dataframe(sentences_df, use_container_width=True)
        
        # Show IDF scores for all terms
        with st.expander("📈 TF-IDF Analysis Details", expanded=False):
            st.subheader("IDF Scores for All Terms")
            st.write("**Higher IDF = More discriminative (appears in fewer sentences)**")
            
            # Calculate document frequency for each term
            term_frequencies = {}
            for term in idf_scores.keys():
                freq = sum(1 for s in doc.sentences if term in s.tokens)
                term_frequencies[term] = freq
            
            idf_data = []
            for term, score in sorted(idf_scores.items(), key=lambda x: x[1], reverse=True):
                idf_data.append({
                    "Term": term,
                    "Frequency": term_frequencies[term],
                    "IDF Score": f"{score:.4f}"
                })
            
            idf_df = pd.DataFrame(idf_data)
            st.dataframe(idf_df, use_container_width=True, height=200)
            
            # Show complete TF-IDF vectors for each sentence
            st.subheader("Complete TF-IDF Vectors by Sentence")
            
            # Create ordered list of all terms for consistent vector representation
            all_terms = sorted(idf_scores.keys())
            
            for i, (sentence, tfidf_vec) in enumerate(zip(doc.sentences, tfidf_vectors)):
                with st.expander(f"Sentence {i+1}: {sentence.text[:60]}..."):
                    if tfidf_vec:
                        # Show table of non-zero terms
                        tfidf_items = sorted(tfidf_vec.items(), key=lambda x: x[1], reverse=True)
                        tfidf_sentence_data = [
                            {"Term": term, "TF-IDF Score": f"{score:.4f}"}
                            for term, score in tfidf_items
                        ]
                        tfidf_sentence_df = pd.DataFrame(tfidf_sentence_data)
                        st.dataframe(tfidf_sentence_df, use_container_width=True)
                        
                        # Show complete vector representation
                        st.write("**Complete TF-IDF Vector:**")
                        vector_values = []
                        for term in all_terms:
                            score = tfidf_vec.get(term, 0.0)
                            vector_values.append(f"{score:.3f}")
                        
                        # Display vector in a readable format
                        vector_str = "[" + ", ".join(vector_values) + "]"
                        st.code(vector_str, language=None)
                        
                        # Show term order for reference
                        st.write("**Term Order:**")
                        terms_str = "[" + ", ".join(all_terms) + "]"
                        st.code(terms_str, language=None)
                    else:
                        st.write("No TF-IDF terms for this sentence")
    
    # Step 2: Feature Extraction
    st.header("📊 Step 2: Feature Extraction (6 Features)")
    with st.expander("Feature Extraction Details", expanded=True):
        st.write("**Running:** Title words, Sentence length, Position, Numerical data, Thematic words, Sentence similarity")
        
        with st.spinner("Extracting features..."):
            feats, simM = extract_features(doc)
        
        st.success("✅ Extracted 6 features for each sentence")
        
        # Create features dataframe
        features_data = []
        for i, f in enumerate(feats):
            features_data.append({
                "Sentence #": i+1,
                "Title Words": f.get("title_words", 0),
                "Sentence Length": f.get("sentence_length", 0),
                "Position Score": f.get("sentence_position", 0),
                "Numerical Data": f.get("numerical_data", 0),
                "Thematic Words": f.get("thematic_words", 0),
                "Similarity Score": f.get("sentence_similarity", 0),
                "Total Score": sum(f.values())
            })
        
        features_df = pd.DataFrame(features_data)
        st.dataframe(features_df, use_container_width=True)
        
        # Show similarity matrix
        st.subheader("Sentence Similarity Matrix")
        n_sentences = len(simM)
        
        if n_sentences <= 50:  # Only show matrix for small documents
            sim_df = pd.DataFrame(simM, 
                                 columns=[f"S{i+1}" for i in range(len(simM))],
                                 index=[f"S{i+1}" for i in range(len(simM))])
            st.dataframe(sim_df, use_container_width=True)
        else:
            # For large documents, show summary statistics instead
            st.info(f"📊 Matrix too large to display ({n_sentences}×{n_sentences} = {n_sentences**2:,} cells)")
            
            # Show similarity statistics
            flat_sim = [simM[i][j] for i in range(n_sentences) for j in range(i+1, n_sentences)]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Similarity", f"{min(flat_sim):.3f}")
            with col2:
                st.metric("Max Similarity", f"{max(flat_sim):.3f}")
            with col3:
                st.metric("Mean Similarity", f"{np.mean(flat_sim):.3f}")
            with col4:
                st.metric("Std Similarity", f"{np.std(flat_sim):.3f}")
            
            # Show top similarities
            st.subheader("Top 10 Most Similar Sentence Pairs")
            top_pairs = []
            for i in range(n_sentences):
                for j in range(i+1, n_sentences):
                    top_pairs.append((i, j, simM[i][j]))
            
            top_pairs.sort(key=lambda x: x[2], reverse=True)
            top_10 = top_pairs[:10]
            
            pairs_data = []
            for i, j, sim in top_10:
                pairs_data.append({
                    "Pair": f"S{i+1} ↔ S{j+1}",
                    "Similarity": f"{sim:.3f}",
                    "Above Threshold": "✅" if sim >= sim_threshold else "❌"
                })
            
            pairs_df = pd.DataFrame(pairs_data)
            st.dataframe(pairs_df, use_container_width=True)
    
    # Step 3: Graph Construction
    st.header("🕸️ Step 3: Graph Construction")
    with st.expander("Graph Construction Details", expanded=True):
        st.write(f"**Running:** Creating nodes (sentences) and edges (similarity ≥ {sim_threshold})")
        
        with st.spinner("Building graph..."):
            graph = build_graph(doc, simM, threshold=sim_threshold)
        
        st.success(f"✅ Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nodes (Sentences)", len(graph.nodes))
        with col2:
            st.metric("Edges", len(graph.edges))
        with col3:
            max_possible_edges = len(graph.nodes) * (len(graph.nodes) - 1) // 2
            density = len(graph.edges) / max_possible_edges if max_possible_edges > 0 else 0
            st.metric("Graph Density", f"{density:.2%}")
        
        # Show edges
        if graph.edges:
            edges_data = []
            for edge in graph.edges:
                edges_data.append({
                    "From": f"S{edge.i+1}",
                    "To": f"S{edge.j+1}",
                    "Weight": f"{edge.weight:.3f}",
                    "Above Threshold": "✅" if edge.weight >= sim_threshold else "❌"
                })
            edges_df = pd.DataFrame(edges_data)
            st.dataframe(edges_df, use_container_width=True)
        else:
            st.warning("No edges found with current similarity threshold")
    
    # Step 4: Triangular Sub-graph Filtering
    st.header("🔺 Step 4: Triangular Sub-graph Filtering")
    with st.expander("Triangle Detection Details", expanded=True):
        st.write("**Running:** Finding triangles using adjacency matrix")
        
        with st.spinner("Finding triangles..."):
            A = build_adjacency(graph)
            triangles = find_triangles(A)
            reduced_graph = reduce_graph_to_triangles(graph, triangles)
        
        st.success(f"✅ Found {len(triangles)} triangles")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Triangles Found", len(triangles))
        with col2:
            st.metric("Edges in Triangles", len(reduced_graph.edges))
        with col3:
            triangle_nodes = set()
            for tri in triangles:
                triangle_nodes.update(tri)
            st.metric("Nodes in Triangles", len(triangle_nodes))
        
        # Graph Visualization
        st.subheader("📊 Graph Visualization")
        if len(graph.nodes) <= 50:  # Only show visualization for reasonably sized graphs
            try:
                with st.spinner("Generating graph visualization..."):
                    graph_image = draw_graph_visualization(graph, triangles, doc, simM, sim_threshold)
                st.image(graph_image, caption="Graph Structure and Triangle Detection", use_column_width=True)
                
                # Add explanation
                st.info("""
                **Left Plot**: Shows all edges above the similarity threshold. Edge thickness represents similarity strength.
                
                **Right Plot**: Highlights detected triangles in different colors. Yellow nodes are part of triangles.
                
                **Triangles**: Groups of 3 sentences that are all mutually similar (form closed loops in the graph).
                """)
            except Exception as e:
                st.error(f"Could not generate graph visualization: {str(e)}")
        else:
            st.info(f"📊 Graph too large to visualize ({len(graph.nodes)} nodes). Showing statistics instead.")
        
        # Show adjacency matrix
        st.subheader("Adjacency Matrix")
        n_nodes = len(A)
        
        if n_nodes <= 50:  # Only show matrix for small documents
            adj_df = pd.DataFrame(A, 
                                 columns=[f"S{i+1}" for i in range(len(A))],
                                 index=[f"S{i+1}" for i in range(len(A))])
            st.dataframe(adj_df, use_container_width=True)
        else:
            # For large documents, show summary instead
            st.info(f"📊 Adjacency matrix too large to display ({n_nodes}×{n_nodes} = {n_nodes**2:,} cells)")
            
            # Show adjacency statistics
            total_connections = sum(sum(row) for row in A) // 2  # Divide by 2 since matrix is symmetric
            max_possible = n_nodes * (n_nodes - 1) // 2
            density = total_connections / max_possible if max_possible > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Edges", total_connections)
            with col2:
                st.metric("Max Possible", max_possible)
            with col3:
                st.metric("Density", f"{density:.2%}")
            
            # Show nodes with most connections
            node_degrees = [sum(A[i]) for i in range(n_nodes)]
            top_nodes = sorted(enumerate(node_degrees), key=lambda x: x[1], reverse=True)[:10]
            
            st.subheader("Top 10 Most Connected Sentences")
            degree_data = []
            for node_idx, degree in top_nodes:
                degree_data.append({
                    "Sentence": f"S{node_idx+1}",
                    "Connections": degree,
                    "Text Preview": doc.sentences[node_idx].text[:80] + "..." if len(doc.sentences[node_idx].text) > 80 else doc.sentences[node_idx].text
                })
            
            degree_df = pd.DataFrame(degree_data)
            st.dataframe(degree_df, use_container_width=True)
        
        # Show triangles
        if triangles:
            triangles_data = []
            for i, (x, y, z) in enumerate(triangles):
                triangles_data.append({
                    "Triangle #": i+1,
                    "Sentences": f"S{x+1}, S{y+1}, S{z+1}",
                    "Weights": f"{simM[x][y]:.3f}, {simM[y][z]:.3f}, {simM[x][z]:.3f}"
                })
            triangles_df = pd.DataFrame(triangles_data)
            st.dataframe(triangles_df, use_container_width=True)
        else:
            st.warning("No triangles found - summary will use fallback method")
    
    # Step 5: Sentence Scoring
    st.header("🎯 Step 5: Sentence Scoring")
    with st.expander("Scoring Details", expanded=True):
        st.write("**Running:** BitVector calculation and final scoring")
        
        with st.spinner("Calculating scores..."):
            scores, bitvec = score_sentences(doc, feats, graph=graph, threshold=sim_threshold)
        
        st.success("✅ Calculated sentence scores")
        
        # Create scoring dataframe
        scoring_data = []
        for i in range(len(doc.sentences)):
            feature_sum = sum(feats[i].values())
            scoring_data.append({
                "Sentence #": i+1,
                "BitVector": bitvec[i],
                "Feature Sum": f"{feature_sum:.3f}",
                "Final Score": f"{scores[i]:.3f}",
                "In Triangle": "✅" if bitvec[i] == 1 else "❌",
                "Text Preview": doc.sentences[i].text[:80] + "..." if len(doc.sentences[i].text) > 80 else doc.sentences[i].text
            })
        
        scoring_df = pd.DataFrame(scoring_data)
        st.dataframe(scoring_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentences in Triangles", sum(bitvec))
        with col2:
            st.metric("Sentences NOT in Triangles", len(bitvec) - sum(bitvec))
    
    # Step 6: Summary Generation
    st.header("📝 Step 6: Summary Generation")
    with st.expander("Summary Generation Details", expanded=True):
        st.write(f"**Running:** Selecting top sentences with compression ratio {compression_ratio}")
        
        with st.spinner("Generating summary..."):
            summary = generate_summary(doc, scores, bitvec, compression_ratio=compression_ratio)
        
        st.success("✅ Summary generated")
        
        # Show selection process
        n = len(doc.sentences)
        k = max(1, int(round(n * compression_ratio)))
        candidates = [(i, scores[i]) for i in range(n) if bitvec[i] == 1]
        
        if len(candidates) < k:
            st.warning("Not enough sentences in triangles - using fallback selection")
            candidates = [(i, scores[i]) for i in range(n)]
        
        selected_indices = [i for i, _ in sorted(candidates, key=lambda x: x[1], reverse=True)[:k]]
        selected_indices.sort()
        
        selection_data = []
        for i in range(len(doc.sentences)):
            selection_data.append({
                "Sentence #": i+1,
                "Score": f"{scores[i]:.3f}",
                "Selected": "✅" if i in selected_indices else "❌",
                "Reason": "Top score in triangles" if i in selected_indices and bitvec[i] == 1 
                         else "Fallback selection" if i in selected_indices 
                         else "Not selected",
                "Text": doc.sentences[i].text
            })
        
        selection_df = pd.DataFrame(selection_data)
        st.dataframe(selection_df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Sentences", k)
        with col2:
            st.metric("Actually Selected", len(selected_indices))
        with col3:
            actual_ratio = len(selected_indices) / len(doc.sentences)
            st.metric("Actual Ratio", f"{actual_ratio:.2%}")
    
    return summary

def main():
    st.title("Triangle-graph Summarizer")
    st.write("Upload a text file to generate a summary using triangle-graph algorithm")
    
    # Sidebar controls
    ratio, threshold, debug_mode = create_sidebar_controls()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a text file", 
        type=['txt', 'rtf', 'md'],
        help="Upload a text file to summarize (supports .txt, .rtf, .md formats)"
    )
    
    if uploaded_file is not None:
        # Load and display original text
        text = load_text_from_file(uploaded_file)
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        st.subheader(f"Original Text ({file_extension.upper()} format)")
        st.text_area("Content", text, height=200, disabled=True)
        
        # Title input section
        st.subheader("Document Title")
        default_title = generate_default_title(uploaded_file.name)
        document_title = st.text_input(
            "Enter document title:",
            value=default_title,
            help="This title will be used for feature extraction. Default is generated from filename."
        )
        
        # Generate summary button
        if st.button("Generate Summary", type="primary"):
            try:
                if debug_mode:
                    # Run with detailed debugging
                    st.markdown("---")
                    st.title("🔍 Pipeline Debug Mode")
                    result = debug_pipeline(text, document_title, ratio, threshold)
                else:
                    # Run simple mode
                    with st.spinner("Generating summary..."):
                        result = summarize(text, title=document_title, compression_ratio=ratio, sim_threshold=threshold)
                
                # Final summary display
                st.markdown("---")
                st.header("📋 Final Summary")
                st.text_area("Generated Summary", result, height=150, disabled=True)
                
                # Display statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Length", len(text.split()))
                with col2:
                    st.metric("Summary Length", len(result.split()) if result else 0)
                with col3:
                    compression = len(result.split()) / len(text.split()) if text and result else 0
                    st.metric("Actual Compression", f"{compression:.2%}")
                    
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
