import os
import shutil
import base64
from math import sqrt
from typing import List, Dict, Any

import openai
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# For TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Optional: for tokenization or cleaning
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import word_tokenize
import string

load_dotenv()

########################################
# Global Config
########################################

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not set in environment variables.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# Download NLTK data if needed
nltk.download('punkt', quiet=True)

########################################
# Data Models
########################################

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]]

########################################
# In-Memory Stores
########################################

document_chunks: List[Dict[str, Any]] = []
# chunk_info: {
#   "id": int,  # index in the global list
#   "filename": str,
#   "page_num": int,
#   "text": str,
#   "embedding": List[float]
# }

tfidf_vectorizer = None
tfidf_matrix = None
all_texts_for_tfidf = []  # store chunk texts for TF-IDF

########################################
# Embeddings & Cosine
########################################

def compute_cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def get_embeddings(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    # openai>=1.0.0 => openai.embeddings.create
    resp = openai.embeddings.create(
        model=model,
        input=[text]
    )
    return resp.data[0].embedding

########################################
# Text Preprocessing
########################################

def preprocess_text_for_tfidf(text: str) -> str:
    """
    Basic text cleaning to feed TF-IDF. 
    You can do advanced cleaning (lowercasing, removing stopwords/punctuation, etc.) 
    but be mindful not to lose crucial policy terms.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return ' '.join(tokens)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            break
    return chunks

def get_page_screenshot(file_path: str, page_num: int, highlight_text: str = None) -> str:
    doc = fitz.open(file_path)
    page = doc[page_num]

    if highlight_text:
        snippet = highlight_text[:50]
        instances = page.search_for(snippet)
        for inst in instances:
            page.add_highlight_annot(inst)

    pix = page.get_pixmap()
    img_data = pix.tobytes("png")
    return base64.b64encode(img_data).decode()

########################################
# Routes
########################################

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    1) Save PDFs
    2) Extract pages -> chunk
    3) For each chunk: 
       - store in document_chunks with an ID
       - store raw text in all_texts_for_tfidf
       - create embedding
    4) Rebuild TF-IDF
    """
    global tfidf_vectorizer, tfidf_matrix

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    start_id = len(document_chunks)
    chunk_id = start_id

    for f in files:
        file_path = os.path.join(upload_dir, f.filename)
        # Save file
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving file {f.filename}: {e}")

        # PyMuPDF open
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not open {f.filename}: {e}")

        # Chunk each page
        for page_idx in range(len(doc)):
            page_text = doc[page_idx].get_text()
            if not page_text.strip():
                continue

            chunks = chunk_text(page_text)
            for c in chunks:
                # For TF-IDF we store the preprocessed text
                preprocessed = preprocess_text_for_tfidf(c)

                # Build embedding 
                try:
                    embedding = get_embeddings(c)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Embedding failed on {f.filename} p{page_idx+1}: {e}")

                doc_info = {
                    "id": chunk_id,
                    "filename": f.filename,
                    "page_num": page_idx,
                    "text": c,
                    "embedding": embedding,
                }
                document_chunks.append(doc_info)
                all_texts_for_tfidf.append(preprocessed)
                chunk_id += 1

    # Rebuild TF-IDF from scratch
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts_for_tfidf)

    return {"message": "Files uploaded, chunked, and processed successfully."}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    1) Retrieve top 10 chunks by TF-IDF
    2) Re-rank those 10 with OpenAI embedding
    3) Keep top 5
    4) Build context, call GPT-4o
    5) Return direct quotes, citations, screenshots
    """
    global tfidf_vectorizer, tfidf_matrix

    if not document_chunks:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    user_query = request.message

    # Step 1: TF-IDF retrieval
    if tfidf_vectorizer is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF not initialized; upload PDFs first.")

    # Preprocess user query for TF-IDF
    query_preproc = preprocess_text_for_tfidf(user_query)
    query_vec = tfidf_vectorizer.transform([query_preproc])  # shape (1, vocab)

    # Compute scores
    scores = tfidf_matrix @ query_vec.T  # shape (num_docs, 1)
    scores_array = scores.toarray().ravel()  # (num_docs,)

    # Sort descending by lexical similarity
    top_lexical_indices = np.argsort(scores_array)[::-1]

    # Keep top 10
    top_lexical_indices = top_lexical_indices[:10]

    # Step 2: Re-rank those 10 by embedding similarity
    # Embed user query with OpenAI
    try:
        user_query_embedding = get_embeddings(user_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed user query: {e}")

    # We'll gather (sim, chunk_info) for these top 10
    re_rank_candidates = []
    for idx in top_lexical_indices:
        chunk_info = document_chunks[idx]
        sim = compute_cosine_similarity(user_query_embedding, chunk_info["embedding"])
        re_rank_candidates.append((sim, chunk_info))

    re_rank_candidates.sort(key=lambda x: x[0], reverse=True)

    # Keep top 5
    top_chunks = re_rank_candidates[:5]

    # Step 3: Build context
    context = ""
    for sim, chunk in top_chunks:
        context += f"File: {chunk['filename']} | Page {chunk['page_num']+1}\n"
        context += chunk["text"] + "\n\n"

    # Step 4: GPT-4o chat
    system_prompt = (
        "You are a hospital policy AI. Use ONLY the provided context to answer the user's question. "
        "Quote directly from the text when possible. If the answer isn't in the context, say you're unsure."
    )

    user_prompt = (
        f"CONTEXT:\n{context}\n"
        f"USER QUESTION:\n{user_query}\n"
    )

    try:
        completion = openai.chat.completions.create(
            model="gpt-4o",  # your custom model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI ChatCompletion error: {e}")

    # Step 5: Citations & Screenshots
    citations = []
    screenshots = []
    for sim, chunk in top_chunks:
        citations.append({
            "policy": chunk["filename"],
            "section": f"Page {chunk['page_num']+1}"
        })
        file_path = os.path.join("uploads", chunk["filename"])
        try:
            screenshot_b64 = get_page_screenshot(file_path, chunk["page_num"], chunk["text"][:50])
            screenshots.append(screenshot_b64)
        except Exception as e:
            print(f"Screenshot error: {e}")

    return {
        "content": answer,
        "citations": citations,
        "screenshots": screenshots
    }


