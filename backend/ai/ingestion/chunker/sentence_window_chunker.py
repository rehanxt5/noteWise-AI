'''
Docstring for backend.ai.ingestion.chunker.sentence_window_chunker

This module provides functionality to chunk text documents into smaller segments based on sentence windows for ingestion into the RAG system.
It includes methods to split text into sentences and group them into chunks of specified sizes.
It overlaps sentences between chunks to maintain context.

Important Note:
- This chunker is designed to work with plain text documents. Ensure that the input text is clean and free from non-text elements.
- This chunker does not perform any OCR or image processing. Use it only with text-based content.

Author: Rehan Akhtar
Date: 30 Jan 2026
'''
import numpy as np
import re

def simple_sentence_window_chunker(text: str, chunk_size: int = 5, overlap: int = 2) -> list:
    '''
    Chunks the input text into smaller segments based on sentence windows. It splits the text based on the '.' , '!' and '?' delimiters.
    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The number of sentences in each chunk.
        overlap (int): The number of overlapping sentences between consecutive chunks.

    Returns:
        numpy.ndarray: An array containing the text chunks.
    '''
    cleanedData = re.sub(r'\\[ntru"\'\\]', ' ', text)  # Remove common escapes
    cleanedData = re.sub(r'\\u[0-9a-fA-F]{4}', '', cleanedData)  # Remove unicode escapes
    cleanedData = re.sub(r'\s+', ' ', cleanedData).strip()  # Normalize whitespace

    sentences = re.split(r'(?<=[.!?])\s+', cleanedData)

    chunks = []
    
    for i in range(0, len(sentences), chunk_size - overlap):
        window = sentences[i:i + chunk_size]

        chunk_text = ' '.join(window)
        chunks.append(chunk_text)

        if len(window)< chunk_size:
            break
    return np.array(chunks)
