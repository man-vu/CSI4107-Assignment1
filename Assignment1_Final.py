# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 00:38:50 2025

@author: TIM
"""

import re
import math
import pandas as pd
import requests

from collections import defaultdict
from nltk.stem import PorterStemmer
import pytrec_eval

# Initialize Porter Stemmer
stemmer = PorterStemmer()


# --------------------- Preprocess and indexing ---------------------------#
#                                                                          #
# --------------------- Preprocess and indexing ---------------------------#

def preprocess_text(text, stopwords):
    """
    This module preprocesses text by:
    1. Tokenizing the text.
    2. Converting all words to lowercase.
    3. Removing stopwords.
    4. Filtering out non-alphanumeric characters.
    5. Applying Porter Stemming.
    """
    # Remove non-text markup (e.g., HTML tags or XML tags)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Tokenize and filter out punctuation and numbers
    tokens = re.findall(r'\b[a-z]+\b', text.lower())

    # Remove stopwords
    filtered_tokens = [token for token in tokens if token not in stopwords]

    # Apply Porter stemming
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    return stemmed_tokens


def preprocess_text_dataframe(df, text_column, stopwords):
    df[text_column + "_tokens"] = df[text_column].apply(lambda x: preprocess_text(x, stopwords))
    return df

def build_inverted_index_with_stats(corpus_df, text_column):
    '''
        Input: Preprocessed tokens from documents.
        Output: 
        - An inverted index stored as a DataFrame
        - A document length dictionary which stores the number of tokens in each document
        - Average document length used for BM25 scoring

    '''
    inverted_index = {}
    doc_lengths = defaultdict(int)

    for _, row in corpus_df.iterrows():
        doc_id = row["_id"]
        tokens = row[text_column] # Get the list of tokens from the document


        # Create a dictionary to store term frequencies in the document
        freq_map = defaultdict(int)
        for token in tokens:
            freq_map[token] += 1

        # store document length which is necessary for BM25 calculation
        doc_lengths[doc_id] = len(tokens)

        # Update the inverted index
        for token, tf in freq_map.items():
            if token not in inverted_index:
                inverted_index[token] = {
                    "df": 0,
                    "postings": {}
                }
            # if doc_id not present, increment df by 1
            if doc_id not in inverted_index[token]["postings"]:
                inverted_index[token]["df"] += 1

            inverted_index[token]["postings"][doc_id] = tf

    # Calculate the average document length (avgdl)
    total_length = sum(doc_lengths.values())
    avgdl = float(total_length) / len(doc_lengths) if doc_lengths else 0.0

    return inverted_index, doc_lengths, avgdl


# --------------------- BM25 ---------------------------
#                                                      #
# --------------------- BM25 ---------------------------

def bm25_score(tf, df, dl, avgdl, N, k1, b):
    """
        Computes the BM25 score based on  
        https://www.site.uottawa.ca/~diana/csi4107/BM25.pdf
    """
    # IDF
    idf = math.log((N - df + 0.5) / (df + 0.5))

    # TF saturation
    numerator = tf * idf * (k1 + 1)
    denominator = k1 * ((1 - b) + b * (dl / avgdl)) + tf

    return (numerator / denominator)

def retrieve_and_rank_bm25(query_tokens, inverted_index, doc_lengths, avgdl, N,
                           k1, b):
    # Create a dictionary to accumulate BM25 scores for each doc_id.
    scores = defaultdict(float)

    # For each query token, count how often each term appears in the query.
    q_freq = defaultdict(int)
    for qt in query_tokens:
        q_freq[qt] += 1

    # This loop calculates BM25 scores for each term in the query.
    for qt, qf in q_freq.items():
        # If the query term is not in the index, skip it.
        if qt not in inverted_index:
            continue

        # Get the document frequency for the term
        df = inverted_index[qt]["df"]
        # Get the postings list (mapping doc_id -> term frequency in that doc)
        postings = inverted_index[qt]["postings"]

        # Calculate BM25 for each document where this term appears
        for doc_id, tf in postings.items():
            dl = doc_lengths[doc_id]  # The length of the current document
            score = bm25_score(tf, df, dl, avgdl, N, k1, b)
            # Multiply by qf to account for repeated query terms (if desired)
            scores[doc_id] += score * qf

    # Sort documents by their accumulated BM25 scores in descending order
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

def generate_results_file_bm25(queries_df, inverted_index, doc_lengths, avgdl,
                               output_file, run_name,
                               k1, b):
    # Number of documents in the collection
    N = len(doc_lengths)

    # Open the file for writing results
    with open(output_file, "w", encoding="utf-8") as f:
        # Iterate over each query
        for _, row in queries_df.iterrows():
            query_id = row["_id"]         # Unique ID for the query
            query_tokens = row["text_tokens"]

            # Retrieve and rank documents using BM25
            ranked_docs = retrieve_and_rank_bm25(
                query_tokens, inverted_index, doc_lengths, avgdl, N, k1, b
            )

            # Keep only the top 100 documents
            top_docs = ranked_docs[:100]

            rank = 1
            for doc_id, score in top_docs:
                line = f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}\n"
                f.write(line)
                rank += 1

# -------------- Evaluation modules -------------------
#                                                     #
# -------------- Evaluation modules -------------------

def load_qrels_tsv(filepath):
    """
    Loads ground truth table
    """
    df = pd.read_csv(filepath, sep="\t", dtype=str)  # Automatically detects header

    # Ensure the column names match expected ones
    expected_cols = {"query-id", "corpus-id", "score"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"TSV file must contain columns: {expected_cols}. Found: {df.columns}")

    results = {}
    for _, row in df.iterrows():
        query_id, doc_id, score = f"q{row['query-id']}", f"d{row['corpus-id']}", int(row["score"])
        
        if query_id not in results:
            results[query_id] = {}
        
        results[query_id][doc_id] = score
    
    return results

def load_trec_results(filepath):
    """
    Loads result files from models
    """
    run = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue  # Skip malformed lines
            
            query_id, _, doc_id, _, score, _ = parts
            score = float(score)  # Convert score to float
            
            query_id = f"q{query_id}"
            doc_id = f"d{doc_id}"

            if query_id not in run:
                run[query_id] = {}
            
            run[query_id][doc_id] = score
    
    return run

# -------------- MAIN SCRIPT ---------------------------
#                                                      #
# -------------- MAIN SCRIPT ---------------------------

# 1. Fetch stopwords
stopwords_url = "https://www.site.uottawa.ca/~diana/csi4107/StopWords"
r = requests.get(stopwords_url)
stopwords = set(r.text.splitlines()) if r.status_code == 200 else set()

# 2. Load corpus & queries
corpus_file_path = "./scifact/corpus.jsonl"
queries_file_path = "./scifact/queries.jsonl"

corpus_df = pd.read_json(corpus_file_path, lines=True) 
queries_df = pd.read_json(queries_file_path, lines=True)  
queries_df = preprocess_text_dataframe(queries_df, "text", stopwords)

# 3. Preprocess
corpus_df = preprocess_text_dataframe(corpus_df, "title", stopwords)
corpus_df = preprocess_text_dataframe(corpus_df, "text", stopwords)

# Create a new column combining title and text
corpus_df["title_text"] = corpus_df["title"] + " " + corpus_df["text"]
corpus_df = preprocess_text_dataframe(corpus_df, "title_text", stopwords)


#---------------- BM25 Run 1 ------------------#

# 4. Build inverted index with stats
inverted_index, doc_lengths, avgdl = build_inverted_index_with_stats(corpus_df, "title_tokens")

# 5. Generate results for each query with BM25
results_file_path = "./Results_TitleOnly.txt"
generate_results_file_bm25(
    queries_df,
    inverted_index,
    doc_lengths,
    avgdl,
    results_file_path,
    run_name="MyBM25Run",
    k1=1.2, b=0.75,  # BM25 parameters
)

print("BM25 results generated at:", results_file_path)

#---------------- BM25 Run 2 ------------------#

# 4. Build inverted index with stats
inverted_index, doc_lengths, avgdl = build_inverted_index_with_stats(corpus_df, "title_text_tokens")

# 5. Generate results for each query with BM25
results_file_path = "./Results_TitleAndText.txt"
generate_results_file_bm25(
    queries_df,
    inverted_index,
    doc_lengths,
    avgdl,
    results_file_path,
    run_name="MyBM25Run",
    k1=1.2, b=0.75,  # BM25 parameters
)

print("BM25 results generated at:", results_file_path)


# -------------- Evaluation Script ---------------------------
#                                                            #
# -------------- Evaluation Script ---------------------------

# Define evaluation metrics
metrics = {
    "map",                 # Mean Average Precision
    "Rprec",               # Precision at R (number of relevant documents)
    "bpref",               # Binary Preference
    "recip_rank",          # Mean Reciprocal Rank (MRR)
    
    # Interpolated Precisions
    "iprec_at_recall_0.00", "iprec_at_recall_0.10", "iprec_at_recall_0.20", 
    "iprec_at_recall_0.30", "iprec_at_recall_0.40", "iprec_at_recall_0.50",
    "iprec_at_recall_0.60", "iprec_at_recall_0.70", "iprec_at_recall_0.80",
    "iprec_at_recall_0.90", "iprec_at_recall_1.00",
    
    # Precision at different document cutoffs
    "P_5", "P_10", "P_15", "P_20", "P_30", "P_100", "P_200", "P_500", "P_1000",
    
    # Number of queries and relevant documents
    "num_q", "num_ret", "num_rel", "num_rel_ret"
}

# Load ground truth relevance judgments (qrels)
qrels = load_qrels_tsv("./scifact/qrels/test.tsv")

# Initialize evaluator
evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)

# ---------------- Load Retrieval Results ---------------- #
# Title-Only Run
results_title_only = load_trec_results("./Results_TitleOnly.txt")

# Title+Text Run
results_title_text = load_trec_results("./Results_TitleAndText.txt")

# ---------------- Evaluate Both Models ---------------- #
eval_title_only = evaluator.evaluate(results_title_only)
eval_title_text = evaluator.evaluate(results_title_text)

# ---------------- Compute Average Scores ---------------- #
def compute_avg_results(eval_results):
    avg_results = {}
    for metric in metrics:
        values = [r[metric] for r in eval_results.values() if metric in r]
        if metric in {"num_q", "num_rel", "num_ret", "num_rel_ret"}:
            avg_results[metric] = sum(values)  # Sum for counts
        else:
            avg_results[metric] = sum(values) / len(values) if values else 0.0  # Mean for other metrics
    return avg_results

# Compute averages
avg_results_title_only = compute_avg_results(eval_title_only)
avg_results_title_text = compute_avg_results(eval_title_text)