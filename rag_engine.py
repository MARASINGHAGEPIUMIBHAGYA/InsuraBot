# monkey patch to avoid torch.classes bug on Streamlit
import types
import sys

import torch
torch.classes = types.SimpleNamespace()
sys.modules["torch.classes"] = torch.classes


import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import streamlit as st

class RAGEngine:
    def __init__(self, pdf_path):
        # Load PDF and split text
        doc = fitz.open(pdf_path)
        self.text_chunks = [page.get_text() for page in doc]

        # Embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(self.text_chunks, show_progress_bar=True)

        # FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))

        # Gemini configuration using Streamlit secrets
        self.api_key = st.secrets["GEMINI_API_KEY"]
        try:
            genai.configure(api_key=self.api_key)
            self.gemini = genai.GenerativeModel("gemini-pro")
        except Exception as e:
            print(f"Gemini setup failed: {e}")
            self.gemini = None

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.text_chunks[i] for i in indices[0]]

    def ask(self, question):
        context = "\n\n".join(self.retrieve(question))

        if self.gemini:
            try:
                prompt = f"""Use the context from the insurance policy to answer the user's question.

Context:
{context[:3000]}

Question:
{question}

Answer:"""
                response = self.gemini.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"(Fallback) Relevant content:\n\n{context[:2000]}"
        else:
            return f"(API not loaded) Relevant content:\n\n{context[:2000]}"
