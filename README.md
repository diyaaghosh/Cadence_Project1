#  Multi-PDF RAG Educational Assistant

An AI-powered educational assistant that uses Retrieval-Augmented Generation (RAG) to answer questions from uploaded PDF documents.

## Features

###  Context-Based Question Answering
- Answers user questions strictly based on the uploaded PDF content.
- Understands the relevant section of the document before generating answers.
- Avoids unrelated information and hallucinations.
- Provides clear, exam-friendly explanations.

###  Source Citation Support
- Shows where the answer comes from.
- Displays:
  - PDF file name
  - Page number
  - Relevant extracted content preview
  - Similarity score

###  Multi-PDF Support
- Upload and process multiple PDF documents.
- Creates embeddings for all documents.
- Retrieves the most relevant information from the entire document collection.

###  Topic-Based Summarization
- Generate summaries for any topic available in the PDFs.
- Summary is created only from document context.
- Includes:
  - Important concepts
  - Definitions
  - Key points
  - Steps and comparisons when available

###  AI Quiz Generator
- Generates MCQ quizzes from PDF content.
- Supports difficulty levels:
  - Easy
  - Medium
  - Hard

Each question contains:
- 4 options
- Correct answer
- Explanation

After attempting a question:
- User can check the answer immediately.
- Shows whether the answer is correct or wrong.
- Provides explanation.

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- Groq LLM
- PyMuPDF

## RAG Pipeline

PDF Upload  
↓  
Text Extraction  
↓  
Chunk Splitting  
↓  
Embedding Generation  
↓  
Vector Storage  
↓  
Similarity Retrieval  
↓  
LLM Response Generation


## Setup

1. Clone repository

2. Install dependencies:

```bash
pip install -r requirements.txt
```
3.Create .env

```bash
GROQ_API_KEY=your_api_key
```
4.Run:
```bash
streamlit run app.py
```
