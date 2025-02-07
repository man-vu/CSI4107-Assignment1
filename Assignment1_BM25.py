import math
from collections import defaultdict

# -------------- BM25 ---------------------

def bm25_score(tf, df, dl, avgdl, N, k1=1.2, b=0.75):
    # IDF
    idf = math.log((N - df + 0.5) / (df + 0.5))

    # TF saturation
    numerator = tf * (k1 + 1)
    denominator = k1 * ((1 - b) + b * (dl / avgdl)) + tf

    return idf * (numerator / denominator)

def retrieve_and_rank_bm25(query_tokens, inverted_index, doc_lengths, avgdl, N,
                           k1=1.2, b=0.75):
    scores = defaultdict(float)

    # query term frequencies (optional for standard short queries)
    q_freq = defaultdict(int)
    for qt in query_tokens:
        q_freq[qt] += 1

    for qt, qf in q_freq.items():
        if qt not in inverted_index:
            continue

        df = inverted_index[qt]["df"]
        postings = inverted_index[qt]["postings"]

        for doc_id, tf in postings.items():
            dl = doc_lengths[doc_id]
            score = bm25_score(tf, df, dl, avgdl, N, k1, b)
            # Multiply by qf if you want to account for repeated query terms
            scores[doc_id] += score * qf

    # sort descending by score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

def generate_results_file_bm25(queries_df, inverted_index, doc_lengths, avgdl,
                               output_file, run_name="run_name",
                               k1=1.2, b=0.75):
    N = len(doc_lengths)
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in queries_df.iterrows():
            query_id = row["_id"]
            query_tokens = row["text_tokens"]

            ranked_docs = retrieve_and_rank_bm25(
                query_tokens, inverted_index, doc_lengths, avgdl, N, k1, b
            )

            top_docs = ranked_docs[:100]  # top-100 only

            rank = 1
            for doc_id, score in top_docs:
                line = f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}\n"
                f.write(line)
                rank += 1


# -------------- BM25 + Top Feedback Loop ---------------------

def retrieve_and_rank_bm25_prf(query_tokens, inverted_index, doc_lengths, avgdl, N,
                           k1=1.2, b=0.75, top_k=5):
    """
    Retrieve documents using BM25 and return top-ranked docs for pseudo-relevance feedback.

    Returns:
    - sorted_docs: List of (doc_id, BM25 score)
    - top_k_docs: List of top-k document IDs (for query expansion)
    """
    scores = defaultdict(float)

    # Query term frequencies
    q_freq = defaultdict(int)
    for qt in query_tokens:
        q_freq[qt] += 1

    for qt, qf in q_freq.items():
        if qt not in inverted_index:
            continue

        df = inverted_index[qt]["df"]
        postings = inverted_index[qt]["postings"]

        for doc_id, tf in postings.items():
            dl = doc_lengths[doc_id]
            score = bm25_score(tf, df, dl, avgdl, N, k1, b)
            scores[doc_id] += score * qf  # Multiply by query term frequency

    # Sort descending by score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Select top-k documents for PRF
    top_k_docs = [doc_id for doc_id, _ in sorted_docs[:top_k]]

    return sorted_docs, top_k_docs


def expand_query_with_prf(query_tokens, top_k_docs, corpus_df, num_expansion_terms=3):
    """
    Expand the query using pseudo-relevance feedback (PRF).
    
    - Extracts top frequent words from the top-k documents.
    - Adds `num_expansion_terms` most frequent terms to the original query.

    Returns:
    - Expanded query tokens.
    """
    term_freq = defaultdict(int)

    # Collect term frequencies from the top-k documents
    for doc_id in top_k_docs:
        doc_text_tokens = corpus_df[corpus_df["_id"] == doc_id]["text_tokens"].values[0]
        for token in doc_text_tokens:
            term_freq[token] += 1

    # Sort terms by frequency and select top `num_expansion_terms`
    sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
    expansion_terms = [term for term, _ in sorted_terms[:num_expansion_terms]]

    # Combine original query with expansion terms
    expanded_query = list(set(query_tokens + expansion_terms))

    return expanded_query


def generate_results_file_bm25_prf(queries_df, inverted_index, doc_lengths, avgdl,
                                   corpus_df, output_file, run_name="run_name",
                                   k1=1.2, b=0.75, top_k=5, num_expansion_terms=3):
    """
    Generate results using BM25 with Pseudo-Relevance Feedback (PRF).

    - First retrieves top-k documents using BM25.
    - Expands the query using PRF.
    - Retrieves documents again using the expanded query.
    - Saves top-100 ranked results.

    """
    N = len(doc_lengths)
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in queries_df.iterrows():
            query_id = row["_id"]
            query_tokens = row["text_tokens"]

            # First BM25 retrieval
            ranked_docs, top_k_docs = retrieve_and_rank_bm25_prf(
                query_tokens, inverted_index, doc_lengths, avgdl, N, k1, b, top_k
            )

            # Expand query using PRF
            expanded_query = expand_query_with_prf(query_tokens, top_k_docs, corpus_df, num_expansion_terms)

            # Second BM25 retrieval with expanded query
            ranked_docs = retrieve_and_rank_bm25(
                expanded_query, inverted_index, doc_lengths, avgdl, N, k1, b
            )

            # Take top-100
            top_docs = ranked_docs[:100]

            rank = 1
            for doc_id, score in top_docs:
                line = f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}\n"
                f.write(line)
                rank += 1

