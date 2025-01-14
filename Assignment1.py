# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:41:25 2025

@author: TIM
"""
import re
import pandas as pd
import requests
import math
from collections import defaultdict

# Define the preprocess_text function
def preprocess_text(text, stopwords):
    """
    Preprocess text by:
    1. Tokenizing the text.
    2. Converting all words to lowercase.
    3. Removing stopwords.
    4. Filtering out non-alphanumeric characters.
    """
    # Remove non-text markup (e.g., HTML tags or XML tags)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Tokenize and filter out punctuation and numbers
    tokens = re.findall(r'\b[a-z]+\b', text.lower())

    # Remove stopwords
    filtered_tokens = [token for token in tokens if token not in stopwords]

    return filtered_tokens

def build_inverted_index(corpus_df, text_column):
    """
    Build an inverted index for the given corpus.

    Parameters:
    - corpus_df (pd.DataFrame): DataFrame containing preprocessed tokens and document IDs.
    - text_column (str): Column containing tokenized text.

    Returns:
    - inverted_index (pd.DataFrame): A DataFrame representing the inverted index.
    """
    # Create a default dictionary to hold lists of document IDs for each token
    inverted_index = defaultdict(list)

    # Iterate through the DataFrame rows
    for _, row in corpus_df.iterrows():
        doc_id = row["_id"]  # Document ID
        tokens = row[text_column]  # List of tokens

        # Populate the inverted index
        for token in set(tokens):  # Use `set` to avoid duplicate entries for the same document
            inverted_index[token].append(doc_id)

    # Convert the dictionary to a DataFrame
    inverted_index_df = pd.DataFrame(
        [(token, doc_ids) for token, doc_ids in inverted_index.items()],
        columns=["Token", "Document_IDs"]
    )
    return inverted_index_df

def compute_cosine_similarity(query_tokens, document_tokens):
    """
    Compute the cosine similarity between query tokens and document tokens.

    Parameters:
    - query_tokens (list): List of tokens from the query.
    - document_tokens (list): List of tokens from the document.

    Returns:
    - similarity (float): Cosine similarity score.
    """
    # Count term frequencies in query and document
    query_tf = defaultdict(int)
    doc_tf = defaultdict(int)

    for token in query_tokens:
        query_tf[token] += 1
    for token in document_tokens:
        doc_tf[token] += 1

    # Compute dot product and magnitudes
    dot_product = sum(query_tf[token] * doc_tf[token] for token in query_tf if token in doc_tf)
    query_magnitude = math.sqrt(sum(val**2 for val in query_tf.values()))
    doc_magnitude = math.sqrt(sum(val**2 for val in doc_tf.values()))

    if query_magnitude == 0 or doc_magnitude == 0:
        return 0.0  # Avoid division by zero

    return dot_product / (query_magnitude * doc_magnitude)

def retrieve_and_rank_documents(query_tokens, inverted_index_df, corpus_df):
    """
    Retrieve documents containing at least one query word using the inverted index,
    compute cosine similarity scores, and rank the documents.

    Parameters:
    - query_tokens (list): Tokens from the query.
    - inverted_index_df (pd.DataFrame): The inverted index DataFrame.
    - corpus_df (pd.DataFrame): DataFrame containing the corpus with tokenized documents.

    Returns:
    - ranked_docs (pd.DataFrame): DataFrame with document IDs and similarity scores, sorted by rank.
    """
    # Step 1: Retrieve documents containing at least one query word
    matched_docs = set()
    for token in query_tokens:
        if token in inverted_index_df["Token"].values:
            # Get documents containing the token
            doc_ids = inverted_index_df[inverted_index_df["Token"] == token]["Document_IDs"].values[0]
            matched_docs.update(doc_ids)

    # Step 2: Compute cosine similarity for the matched documents
    similarities = []
    for doc_id in matched_docs:
        # Get the document tokens from the corpus
        document_tokens = corpus_df[corpus_df["_id"] == doc_id]["text_tokens"].values[0]
        
        # Compute cosine similarity
        similarity = compute_cosine_similarity(query_tokens, document_tokens)
        similarities.append((doc_id, similarity))

    # Step 3: Rank documents by similarity scores
    ranked_docs = pd.DataFrame(similarities, columns=["Document_ID", "Similarity"])
    ranked_docs = ranked_docs.sort_values(by="Similarity", ascending=False).reset_index(drop=True)

    return ranked_docs

def generate_results_file(queries_df, inverted_index_df, corpus_df, output_file, run_name="run_name"):
    """
    Generate a results file with similarity scores for all queries.

    Parameters:
    - queries_df (pd.DataFrame): DataFrame containing queries with tokenized text.
    - inverted_index_df (pd.DataFrame): The inverted index DataFrame.
    - corpus_df (pd.DataFrame): DataFrame containing the corpus with tokenized documents.
    - output_file (str): Path to the output file.
    - run_name (str): Tag to use for the run.

    Output:
    - Writes the results to the specified output file.
    """
    results = []

    for query_idx, query_row in queries_df.iterrows():
        query_id = query_row["_id"]
        query_tokens = query_row["text_tokens"]

        # Retrieve and rank documents for the query
        ranked_docs = retrieve_and_rank_documents(query_tokens, inverted_index_df, corpus_df)

        # Add top-1000 results to the results list
        for rank, row in ranked_docs.head(1000).iterrows():
            results.append([
                "Q" + str(query_id), row["Document_ID"], rank + 1, row["Similarity"], run_name
            ])

    # Convert results to DataFrame and save to file
    results_df = pd.DataFrame(results, columns=["query_id", "doc_id", "rank", "score", "tag"])
    results_df.to_csv(output_file, sep=" ", index=False, header=False)


# Fetch the stopwords list from the provided URL
def fetch_stopwords(url):
    response = requests.get(url)
    if response.status_code == 200:
        return set(response.text.splitlines())
    else:
        raise Exception(f"Failed to fetch stopwords from {url}")

# Load corpus.jsonl and queries.jsonl using pandas
def load_jsonl_to_dataframe(filepath):
    return pd.read_json(filepath, lines=True)

# Preprocessing function for DataFrame
def preprocess_text_dataframe(df, text_column, stopwords):
    df[text_column + "_tokens"] = df[text_column].apply(lambda x: preprocess_text(x, stopwords))
    return df

# URL for the stopwords list
stopwords_url = "https://www.site.uottawa.ca/~diana/csi4107/StopWords"
stopwords = fetch_stopwords(stopwords_url)

# Paths to the corpus and queries
corpus_file_path = "./scifact/corpus.jsonl"
queries_file_path = "./scifact/queries.jsonl"

# Load the files into DataFrames
corpus_df = load_jsonl_to_dataframe(corpus_file_path)
queries_df = load_jsonl_to_dataframe(queries_file_path)

# Preprocess corpus and queries
preprocessed_corpus_df = preprocess_text_dataframe(corpus_df, "text", stopwords)
preprocessed_queries_df = preprocess_text_dataframe(queries_df, "text", stopwords)

# Show a preview of the preprocessed data
preprocessed_corpus_df.head(1), preprocessed_queries_df.head(1)

# Build the inverted index
inverted_index_df = build_inverted_index(preprocessed_corpus_df, "text_tokens")

# Specify output file path
results_file_path = "./Results.txt"

# Generate the results file
generate_results_file(preprocessed_queries_df, inverted_index_df, preprocessed_corpus_df, results_file_path)

# Confirm file generation
print(f"Results file generated at: {results_file_path}")
