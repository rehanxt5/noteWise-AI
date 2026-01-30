'''
Docstring for backend.ai.retrieval.retriever

This module provides the functionality to search through the ingested document chunks

'''

import numpy as np

def get_top_k_dense(q_dense , doc_dense , k=50):
    '''
    Retrieves the top-k most similar document chunks based on dense vector representations.

    Args:
        q_dense (numpy.ndarray): The dense vector representation of the query.
        doc_dense (numpy.ndarray): The dense vector representations of the document chunks.
        k (int): The number of top similar chunks to retrieve.
    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Indices of the top-k most similar document chunks.
            - numpy.ndarray: Similarity scores of the top-k document chunks.
    '''
    scores = np.dot(doc_dense, q_dense.T).squeeze()
    top_k_indices = np.argsort(scores)[::-1][:k]
    return top_k_indices,scores[top_k_indices]

def get_top_k_sparse(model,q_sparse , doc_sparse , k=50):
    '''
    Retrieves the top-k most similar document chunks based on sparse vector representations.

    Args:
        model: The model used to compute similarity scores between sparse vectors. (BGE-M3)
        q_sparse (numpy.ndarray): The sparse vector representation of the query.
        doc_sparse (numpy.ndarray): The sparse vector representations of the document chunks.
        k (int): The number of top similar chunks to retrieve.
    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Indices of the top-k most similar document chunks.
            - numpy.ndarray: Similarity scores of the top-k document chunks.
    '''
    q_lexical = q_sparse.squeeze()
    scores = []

    for doc_lexical in doc_sparse:
        score = model.compute_lexical_matching_score(q_lexical, doc_lexical)
        scores.append(score)
    scores = np.array(scores)
    top_k_indices = np.argsort(scores)[::-1][:k]
    return top_k_indices,scores[top_k_indices]

def rrf_fusion(dense_indices , sparse_indices , k =60):
    '''
    Combines the results from dense and sparse retrieval methods using Reciprocal Rank Fusion (RRF).

    Args:
        dense_indices (numpy.ndarray): Indices of the top-k most similar document chunks.
        sparse_indices (numpy.ndarray): Indices of the top-k most similar document chunks.
        k (int): 60 , a constant used in the RRF formula to dampen the influence of lower-ranked documents.
    Returns:
        list: A list of tuples containing document chunk indices and their merged scores, sorted in descending order of scores.
    '''
    merged_scores = {}

    # process dense indices
    for rank, idx in enumerate(dense_indices):
        if idx not in merged_scores:
            merged_scores[idx] = 0
        merged_scores[idx] += 1 / (rank + 1 + 60)
    # process sparse indices
    for rank, idx in enumerate(sparse_indices):
        if idx not in merged_scores:
            merged_scores[idx] = 0
        merged_scores[idx] += 1 / (rank + 1 + 60)
    
    # sort by merged scores
    sorted_results = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results

def hybrid_retrieval(model, q_dense, doc_dense, q_sparse, doc_sparse,search_k=50,top_k=30, dense_weight = 0.7 , sparse_weight=0.3):
    '''
    Performs hybrid retrieval by combining dense and sparse retrieval methods.

    Args:
        model: The model used to compute similarity scores between sparse vectors. (BGE-M3)
        q_dense (numpy.ndarray): The dense vector representation of the query.
        doc_dense (numpy.ndarray): The dense vector representations of the document chunks.
        q_sparse (numpy.ndarray): The sparse vector representation of the query.
        doc_sparse (numpy.ndarray): The sparse vector representations of the document chunks.
        search_k (int): The number of top similar chunks to retrieve from each method.
        top_k (int): The number of top similar chunks to return after fusion.
        dense_weight (float): Weight for dense retrieval scores.
        sparse_weight (float): Weight for sparse retrieval scores.  
    Returns:
        list: A list of tuples containing document chunk indices and their merged scores, sorted in descending order of scores.
    '''
    dense_indices, dense_scores = get_top_k_dense(q_dense, doc_dense, k=search_k)
    sparse_indices, sparse_scores = get_top_k_sparse(model, q_sparse, doc_sparse, k=search_k)

    rrf_results = rrf_fusion(dense_indices, sparse_indices, k=60)
    final_results = rrf_results[:top_k]

    return final_results
