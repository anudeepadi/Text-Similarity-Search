import streamlit as st
import pinecone
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Text Similarity Search")
st.write("Welcome! Please enter your query and select the similarity measure.")

df = pd.read_csv(r"Hydra-Movie-Scrape.csv")
df = df.dropna()
df = df.reset_index(drop=True)
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
q1 = np.load(r"vectors.npy")

pinecone.init(api_key="your_api_key",environment="env_name")
index_name = pinecone.Index("index_name")

def similarity_search_Cos_Sim(inp, k):
    """
    Performs cosine similarity search on the given input.

    Args:
    inp: The input string to search for.
    k: The number of results to return.

    Returns:
    A list of tuples, where each tuple contains the title and summary of the
    top-k most similar documents.
    """

    if len(inp) < 2:
        return []

    # Encode the input query.
    query_vector = model.encode([inp])

    # Calculate the cosine similarity between the query vector and all other
    # vectors in the dataset.
    similarity_scores = cosine_similarity(query_vector.reshape(1, -1), q1).flatten()

    # Sort the similarity scores in descending order.
    sorted_indices = similarity_scores.argsort()[::-1]

    # Get the titles and summaries of the top-k most similar documents.
    matching_data_title = [df['Title'].tolist()[i] for i in sorted_indices[:k]]
    matching_data_summary = [df['Summary'].tolist()[i] for i in sorted_indices[:k]]

    return matching_data_title, matching_data_summary


def similarity_search_pinecone(inp, k):
    """
    Performs similarity search on Pinecone for the given input.

    Args:
        inp: The input string to search for.
        k: The number of results to return.

    Returns:
        A list of tuples, where each tuple contains the title and summary of the
        top-k most similar documents.
    """

    if len(inp) < 2:
        return []

    # Encode the input query.
    query_vector = model.encode(inp).tolist()

    # Perform a similarity search on Pinecone.
    res = index_name.query(query_vector, top_k=k, include_metadata=True)

    # Get the titles and summaries of the top-k most similar documents.
    matching_data_title = []
    matching_data_summary = []
    for x in res['matches']:
        matching_data_title.append(x['metadata']['Title'])
        matching_data_summary.append(x['metadata']['Summary'])

    return matching_data_title, matching_data_summary

query = st.text_input("Enter your query:")
similarity_measure = st.radio("Select a similarity measure:", ("Cosine Similarity", "Pinecone Similarity"))


if st.button("Search"):
    if similarity_measure == "Cosine Similarity":
      results_title, results_summary = similarity_search_Cos_Sim(query, 5)
      if not results_title:
        st.write("No matching documents found.")
      else:
         for i in range(len(results_title)):
            st.write(f"Title: {results_title[i]}")
            st.write(f"Summary: {results_summary[i]}")
            st.write("---")
    else:
        results_title, results_summary = similarity_search_pinecone(query, 5)
        if not results_title:
            st.write("No matching documents found.")
        else:
            for i in range(len(results_title)):
                st.write(f"Title: {results_title[i]}")
                st.write(f"Summary: {results_summary[i]}")
                st.write("---")


