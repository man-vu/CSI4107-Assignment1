import math
from collections import defaultdict


# ----------------------------------------------------------------------------
#                   TF–IDF + Cosine Similarity
# ----------------------------------------------------------------------------

def build_tfidf_index(corpus_df, text_column):
    """
    Build a TF–IDF index for cosine-similarity retrieval.

    :param corpus_df: DataFrame containing documents.
    :param text_column: Name of the column with preprocessed tokens (e.g., "text_tokens").
    :return:
        doc_vectors: A dict mapping doc_id -> { token: tfidf_weight, ... }
        doc_norms: A dict mapping doc_id -> Euclidean norm of its TF–IDF vector
        idf_dict: A dict mapping token -> IDF value
        N: Total number of documents
    """

    # Number of documents
    N = len(corpus_df)

    # 1. Compute document frequency for each token
    df_counts = defaultdict(int)  # token -> document frequency
    for _, row in corpus_df.iterrows():
        doc_id = row["_id"]
        tokens = set(row[text_column])  # unique tokens for DF
        for token in tokens:
            df_counts[token] += 1

    # 2. Compute IDF for each token
    #    Use standard formula: idf = log(N / df)
    #    (Some libraries use 1 + log(...) or log((N + 1) / df), etc. You can adapt.)
    idf_dict = {}
    for token, df_val in df_counts.items():
        idf_dict[token] = math.log((N / float(df_val)), 10)  # base-10 or natural log

    # 3. Build TF–IDF vectors for each document
    doc_vectors = defaultdict(dict)  # doc_id -> {token: tfidf_weight, ...}

    for _, row in corpus_df.iterrows():
        doc_id = row["_id"]
        tokens = row[text_column]

        # Count frequencies in this doc
        freq_map = defaultdict(int)
        for token in tokens:
            freq_map[token] += 1

        # Compute TF–IDF for each token in this doc
        for token, tf in freq_map.items():
            # Typical TF weighting: 1 + log10(tf)
            tf_weight = 1 + math.log(tf, 10)
            tfidf_weight = tf_weight * idf_dict[token]
            doc_vectors[doc_id][token] = tfidf_weight

    # 4. Compute document norms for cosine normalization
    doc_norms = {}
    for doc_id, token_weights in doc_vectors.items():
        # norm = sqrt( sum of (weight^2) )
        squared_sum = sum((weight ** 2) for weight in token_weights.values())
        doc_norms[doc_id] = math.sqrt(squared_sum)

    return doc_vectors, doc_norms, idf_dict, N


def compute_query_vector(query_tokens, idf_dict):
    """
    Given query tokens, compute its TF–IDF vector (as a dict: token -> weight).
    Uses the same (1 + log10(tf)) * idf for the query side.
    """
    freq_map = defaultdict(int)
    for token in query_tokens:
        freq_map[token] += 1

    query_vector = {}
    for token, tf in freq_map.items():
        if token in idf_dict:
            # Compute TF–IDF for query tokens
            tf_weight = 1 + math.log(tf, 10)
            query_vector[token] = tf_weight * idf_dict[token]

    return query_vector


def compute_vector_norm(vector_dict):
    """
    Euclidean norm for a dict of token->weight.
    """
    squared_sum = sum((w ** 2) for w in vector_dict.values())
    return math.sqrt(squared_sum)


def retrieve_and_rank_cosine(query_tokens, doc_vectors, doc_norms, idf_dict, N, top_k=None):
    """
    Retrieve and rank documents using TF–IDF + Cosine Similarity.

    :param query_tokens: preprocessed tokens of the query
    :param doc_vectors: dict[doc_id -> {token: weight, ...}]
    :param doc_norms: dict[doc_id -> doc_norm]
    :param idf_dict: dict[token -> IDF]
    :param N: total number of documents (not always needed directly here)
    :param top_k: If specified, return the top_k docs; otherwise, return them all.
    :return:
        sorted_docs: list of tuples (doc_id, cosine_score), sorted desc
    """
    # 1. Build query vector
    query_vector = compute_query_vector(query_tokens, idf_dict)
    query_norm = compute_vector_norm(query_vector)

    scores = defaultdict(float)

    # 2. For each token in the query, accumulate partial scores for docs that contain it
    for token, q_weight in query_vector.items():
        # For each doc that has this token
        for doc_id, doc_token_weights in doc_vectors.items():
            if token in doc_token_weights:
                scores[doc_id] += doc_token_weights[token] * q_weight

    # 3. Normalize by product of norms => final cosine similarity
    #    cos_sim = (doc ⋅ query) / (||doc|| * ||query||)
    sorted_docs = []
    for doc_id, dot_product in scores.items():
        denom = doc_norms[doc_id] * query_norm
        if denom != 0:
            score = dot_product / denom
        else:
            score = 0.0
        sorted_docs.append((doc_id, score))

    # 4. Sort descending
    sorted_docs = sorted(sorted_docs, key=lambda x: x[1], reverse=True)

    # If top_k requested, slice
    if top_k is not None:
        sorted_docs = sorted_docs[:top_k]

    return sorted_docs


def generate_results_file_cosine(queries_df,
                                 doc_vectors,
                                 doc_norms,
                                 idf_dict,
                                 N,
                                 output_file,
                                 run_name="run_name",
                                 top_k=100):
    """
    Generates a TREC-style results file using TF–IDF + Cosine Similarity.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in queries_df.iterrows():
            query_id = row["_id"]
            query_tokens = row["text_tokens"]

            ranked_docs = retrieve_and_rank_cosine(
                query_tokens, doc_vectors, doc_norms, idf_dict, N
            )

            # take top-`top_k`
            top_docs = ranked_docs[:top_k]

            rank = 1
            for doc_id, score in top_docs:
                line = f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}\n"
                f.write(line)
                rank += 1


# ----------------------------------------------------------------------------
#                   Pseudo-Relevance Feedback (PRF)
# ----------------------------------------------------------------------------

def retrieve_and_rank_cosine_prf(query_tokens,
                                 doc_vectors,
                                 doc_norms,
                                 idf_dict,
                                 N,
                                 top_k=5):
    """
    Retrieve documents using cosine similarity and also return top_k doc IDs for PRF.
    """
    scores = defaultdict(float)

    # Build query vector
    query_vector = compute_query_vector(query_tokens, idf_dict)
    query_norm = compute_vector_norm(query_vector)

    # Accumulate scores
    for token, q_weight in query_vector.items():
        for doc_id, doc_token_weights in doc_vectors.items():
            if token in doc_token_weights:
                scores[doc_id] += doc_token_weights[token] * q_weight

    # Normalize & sort
    ranked_docs = []
    for doc_id, dot_product in scores.items():
        denom = doc_norms[doc_id] * query_norm
        if denom != 0:
            score = dot_product / denom
        else:
            score = 0.0
        ranked_docs.append((doc_id, score))

    ranked_docs = sorted(ranked_docs, key=lambda x: x[1], reverse=True)
    top_k_docs = [doc_id for (doc_id, _) in ranked_docs[:top_k]]

    return ranked_docs, top_k_docs


def expand_query_with_prf(query_tokens, top_k_docs, corpus_df, num_expansion_terms=3):
    """
    Expand the query using pseudo-relevance feedback (PRF).
    
    - Extract top frequent words from the top-k documents.
    - Add `num_expansion_terms` most frequent terms to the original query.
    """
    term_freq = defaultdict(int)

    # Collect term frequencies from the top-k documents
    for doc_id in top_k_docs:
        doc_text_tokens = corpus_df[corpus_df["_id"] == doc_id]["text_tokens"].values[0]
        for token in doc_text_tokens:
            term_freq[token] += 1

    # Sort by frequency desc
    sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)

    # Pick expansion terms
    expansion_terms = [term for term, _ in sorted_terms[:num_expansion_terms]]

    # Combine original query with expansion terms
    expanded_query = list(set(query_tokens + expansion_terms))

    return expanded_query


def generate_results_file_cosine_prf(queries_df,
                                     doc_vectors,
                                     doc_norms,
                                     idf_dict,
                                     corpus_df,
                                     N,
                                     output_file,
                                     run_name="run_name",
                                     top_k=5,
                                     num_expansion_terms=3,
                                     final_top_k=100):
    """
    Generate results using TF–IDF cosine similarity with Pseudo-Relevance Feedback (PRF):
    1. Retrieve top-k docs with initial query.
    2. Expand query with PRF.
    3. Retrieve again with expanded query.
    4. Output top-`final_top_k` results.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in queries_df.iterrows():
            query_id = row["_id"]
            query_tokens = row["text_tokens"]

            # First retrieval
            ranked_docs, top_k_docs = retrieve_and_rank_cosine_prf(
                query_tokens, doc_vectors, doc_norms, idf_dict, N, top_k
            )

            # Expand query
            expanded_query = expand_query_with_prf(
                query_tokens, top_k_docs, corpus_df, num_expansion_terms
            )

            # Second retrieval with expanded query
            final_ranked_docs = retrieve_and_rank_cosine(
                expanded_query, doc_vectors, doc_norms, idf_dict, N
            )

            # Take top-`final_top_k`
            top_docs = final_ranked_docs[:final_top_k]

            rank = 1
            for doc_id, score in top_docs:
                line = f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}\n"
                f.write(line)
                rank += 1

