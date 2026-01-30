'''
Docstring for backend.ai.ingestion.parser.textParser

This module provides functionality to parse text documents for ingestion into the RAG system.
It includes methods to read text files, clean the content, and prepare it for further processing.

This module can only handle pdf and text files. And will only extract text from these files. Not images,Tables , Charts etc.

Author: Rehan Akhtar
Date: 30 Jan 2026
'''
import mimetypes
from PyPDF2 import PdfReader

def parseFile(filePath:str) -> str:
    '''
    Parses a text or PDF file and returns its content as a string.

    Args:
        filePath (str): The path to the file to be parsed.
    Returns:
        str: The extracted text content from the file.
    Raises:
        ValueError: If the file type is unsupported.
    '''
    mimeType,_ = mimetypes.guess_type(filePath)
    if mimeType == 'text/plain':
        with open(filePath, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    elif mimeType == 'application/pdf':
        with open(filePath, 'rb') as file:
            reader = PdfReader(file)
            content = ''
            for page in reader.pages:
                content += page.extract_text() + '\n'
        return content
    else:
        raise ValueError(f"Unsupported file type: {mimeType}")