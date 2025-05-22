import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# [Previous imports and configuration remain the same until RAGEngine class]

class RAGEngine:
    def __init__(self, pdf_path):
        from sentence_transformers import SentenceTransformer
        import fitz
        import numpy as np
        import faiss
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        doc = fitz.open(pdf_path)
        self.text_chunks = [page.get_text() for page in doc]
        self.embeddings = self.model.encode(self.text_chunks, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        
        # Initialize API client
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.gemini = genai.GenerativeModel("models/gemini-1.0-pro")
            except:
                self.gemini = None
        else:
            self.gemini = None

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.text_chunks[i] for i in indices[0]]

    def ask(self, question):
        context = "\n\n".join(self.retrieve(question))
        
        if self.gemini:
            try:
                prompt = f"""Use this insurance policy context to answer the question:
                
                Context:
                {context[:3000]}
                
                Question: {question}
                """
                response = self.gemini.generate_content(prompt)
                return response.text
            except:
                # Fallback to simple response if API fails
                return f"Here's relevant policy information:\n\n{context[:2000]}..."
        else:
            return f"Based on the policy document:\n\n{context[:2000]}..."