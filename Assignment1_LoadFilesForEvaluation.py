# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:31:36 2025

@author: TIM
"""

import pandas as pd

# -------------- SCRIPTS TO LOAD test.tsv and Results.txt-------------------

def load_qrels_tsv(filepath):
    """
    Load a TREC-formatted TSV file into a dictionary for pytrec_eval,
    prefixing queries with 'q' and documents with 'd'.
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
    Load TREC-formatted results file into a dictionary for pytrec_eval,
    prefixing queries with 'q' and documents with 'd'.
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

# Define all evaluation metrics
metrics = {
    "map",                 # Mean Average Precision
    # "gm_map",              # Geometric Mean Average Precision
    "Rprec",               # Precision at R (number of relevant documents)
    "bpref",               # Binary Preference
    "recip_rank",          # Mean Reciprocal Rank (MRR)
    
    # Interpolated Precision at different recall levels
    "iprec_at_recall_0.00", "iprec_at_recall_0.10", "iprec_at_recall_0.20", 
    "iprec_at_recall_0.30", "iprec_at_recall_0.40", "iprec_at_recall_0.50",
    "iprec_at_recall_0.60", "iprec_at_recall_0.70", "iprec_at_recall_0.80",
    "iprec_at_recall_0.90", "iprec_at_recall_1.00",
    
    # Precision at different document cutoffs
    "P_5", "P_10", "P_15", "P_20", "P_30", "P_100", "P_200", "P_500", "P_1000",
    
    # Number of queries and relevant documents
    "num_q", "num_ret", "num_rel", "num_rel_ret"
}