import re
import math
import pandas as pd
import requests

from Assignment1_LoadFilesForEvaluation import load_qrels_tsv, load_trec_results, metrics
from Assignment1_BM25 import generate_results_file_bm25, generate_results_file_bm25_prf
from Assignment1_Cosine import build_tfidf_index, generate_results_file_cosine, generate_results_file_cosine_prf

from collections import defaultdict
from nltk.stem import PorterStemmer
import pytrec_eval
import matplotlib.pyplot as plt

# Initialize Porter Stemmer
stemmer = PorterStemmer()

def preprocess_text(text, stopwords):
    """
    Preprocess text by:
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


def load_jsonl_to_dataframe(filepath):
    return pd.read_json(filepath, lines=True)

def preprocess_text_dataframe(df, text_column, stopwords):
    df[text_column + "_tokens"] = df[text_column].apply(lambda x: preprocess_text(x, stopwords))
    return df

def build_inverted_index_with_stats(corpus_df, text_column):
    inverted_index = {}
    doc_lengths = defaultdict(int)

    for _, row in corpus_df.iterrows():
        doc_id = row["_id"]
        tokens = row[text_column]

        freq_map = defaultdict(int)
        for token in tokens:
            freq_map[token] += 1

        doc_lengths[doc_id] = len(tokens)

        for token, tf in freq_map.items():
            if token not in inverted_index:
                inverted_index[token] = {
                    "df": 0,
                    "postings": {}
                }
            # if doc_id not present, increment df
            if doc_id not in inverted_index[token]["postings"]:
                inverted_index[token]["df"] += 1

            inverted_index[token]["postings"][doc_id] = tf

    total_length = sum(doc_lengths.values())
    avgdl = float(total_length) / len(doc_lengths) if doc_lengths else 0.0

    return inverted_index, doc_lengths, avgdl


# -------------- MAIN SCRIPT EXAMPLE -------------------
# Define all results files
results_files = {
    "Cosine_PRF": "./Results_Cosine_PRF.txt",
    "Cosine": "./Results_Cosine.txt",
    "BM25_PRF": "./Results_BM25_PRF.txt",
    "BM25": "./Results_BM25.txt",
}

# BM25 parameters 
k1 = 1.2
b=0.6
top_k = 20
num_expansion_terms=3

# 1. Fetch stopwords
stopwords_url = "https://www.site.uottawa.ca/~diana/csi4107/StopWords"
r = requests.get(stopwords_url)
stopwords = set(r.text.splitlines()) if r.status_code == 200 else set()

# 2. Load corpus & queries
corpus_file_path = "./scifact/corpus.jsonl"
queries_file_path = "./scifact/queries.jsonl"

corpus_df = load_jsonl_to_dataframe(corpus_file_path)
queries_df = load_jsonl_to_dataframe(queries_file_path)

# 3. Preprocess
corpus_df = preprocess_text_dataframe(corpus_df, "text", stopwords)
queries_df = preprocess_text_dataframe(queries_df, "text", stopwords)

#---------------- Cosine approach -------

# 4. Build TF–IDF index
doc_vectors, doc_norms, idf_dict, N = build_tfidf_index(corpus_df, "text_tokens")

# 5. Generate standard Cosine Similarity results
results_file_path = results_files["Cosine"]
generate_results_file_cosine(
    queries_df,
    doc_vectors,
    doc_norms,
    idf_dict,
    N,
    results_file_path,
    run_name="MyCosineRun",
    top_k=100
)
print("Cosine results generated at:", results_file_path)

# 6. Generate Cosine Similarity + PRF results
results_file_path_prf = results_files["Cosine_PRF"]
generate_results_file_cosine_prf(
    queries_df,
    doc_vectors,
    doc_norms,
    idf_dict,
    corpus_df,
    N,
    results_file_path_prf,
    run_name="MyCosinePRF",
    top_k=top_k,              # # docs used for PRF
    num_expansion_terms=num_expansion_terms, # expansion terms
    final_top_k=100
)
print("Cosine + PRF results generated at:", results_file_path_prf)

#---------------- BM25 ------------------

# 4. Build inverted index with stats
inverted_index, doc_lengths, avgdl = build_inverted_index_with_stats(corpus_df, "text_tokens")

# 5. Generate results for each query with BM25
results_file_path = results_files["BM25"]
generate_results_file_bm25(
    queries_df,
    inverted_index,
    doc_lengths,
    avgdl,
    results_file_path,
    run_name="MyBM25Run",
    k1=k1, b=b,  # BM25 parameters
)

print("BM25 results generated at:", results_file_path)

# Generate results for each query with BM25 + PRF
results_file_path = results_files["BM25_PRF"]
generate_results_file_bm25_prf(
    queries_df,
    inverted_index,
    doc_lengths,
    avgdl,
    corpus_df,
    results_file_path,
    run_name="MyBM25PRF",
    k1=k1, b=b,  # BM25 parameters
    top_k=top_k,         # Number of top-ranked docs used for PRF
    num_expansion_terms=num_expansion_terms  # Number of expansion terms added
)

print("BM25 + PRF results generated at:", results_file_path)

#------------------ Evaluation Script ----------------------#

# Load test.tsv (ground truth qrels)
qrels = load_qrels_tsv("./scifact/qrels/test.tsv")

# Initialize evaluator
evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)

# Dictionary to store evaluation results
all_results = {}

for model_name, file_path in results_files.items():
    print(f"Evaluating: {model_name}")

    # Load retrieval results
    model_results = load_trec_results(file_path)

    # Evaluate results
    evaluation_results = evaluator.evaluate(model_results)

    # Compute average scores across all queries
    average_results = {}

    for metric in metrics:
        values = [r[metric] for r in evaluation_results.values() if metric in r]

        if metric == "gm_map":
            # Geometric mean calculation
            values = [math.log(v) for v in values if v > 0]  # Avoid log(0)
            average_results[metric] = math.exp(sum(values) / len(values)) if values else 0.0
        elif metric in {"num_q", "num_rel", "num_ret", "num_rel_ret"}:
            # Summing total counts
            average_results[metric] = sum(values)
        else:
            # Standard arithmetic mean for everything else
            average_results[metric] = sum(values) / len(values) if values else 0.0

    # Store results
    all_results[model_name] = average_results

# Print all results in a structured format
for model, metrics_data in all_results.items():
    print(f"\n=== Results for {model} ===")
    for metric, value in metrics_data.items():
        print(f"{metric}: {value:.4f}")


# Convert all_results into a DataFrame for easy plotting
df_results = pd.DataFrame(all_results).T  # Transpose to get models as rows

# Select relevant metrics for visualization
plot_metrics = ["map", "Rprec", "bpref", "recip_rank"]

# Plot each metric
for metric in plot_metrics:
    plt.figure(figsize=(8, 5))
    df_results[metric].plot(kind="bar", rot=0, legend=True)
    plt.title(f"Comparison of Models - {metric}")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()