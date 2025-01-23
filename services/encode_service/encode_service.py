from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
from dataclasses import dataclass
import logging
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path

# Define the response structure
@dataclass
class EncodeResponse:
    success: bool
    message: str
    timestamp: datetime
    documents_count: int
    error: Optional[str] = None

class DocumentStore:
    def __init__(self):
        self.documents: Dict[str, dict] = {}
        self.embeddings_file = "embeddings.json"
        self._load_embeddings()

    def _load_embeddings(self):
        """Load existing embeddings from file"""
        try:
            if Path(self.embeddings_file).exists():
                with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for doc in data:
                        doc['embedding'] = np.array(doc['embedding'])
                        self.documents[doc['id']] = doc
        except Exception as e:
            logging.error(f"Error loading embeddings: {str(e)}")

    def _save_embeddings(self):
        """Save embeddings to file"""
        try:
            serializable_docs = []
            for doc in self.documents.values():
                doc_copy = doc.copy()
                doc_copy['embedding'] = doc_copy['embedding'].tolist()
                serializable_docs.append(doc_copy)
                
            with open(self.embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving embeddings: {str(e)}")

    def add_document(self, doc_id: str, text: str, embedding: np.ndarray):
        """Add a document with its embedding"""
        self.documents[doc_id] = {
            'id': doc_id,
            'text': text,
            'embedding': embedding,
            'timestamp': datetime.now().isoformat()
        }
        self._save_embeddings()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """Search for similar documents using cosine similarity"""
        if not self.documents:
            return []

        results = []
        for doc in self.documents.values():
            similarity = self._cosine_similarity(query_embedding, doc['embedding'])
            results.append({
                'text': doc['text'],
                'similarity_score': float(similarity)
            })

        # Sort by similarity score and return top_k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)

# Class for the encoding service
class EncodeService:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.document_store = DocumentStore()
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def get_encoding_status(self) -> bool:
        """Check if the encoding service is available and running"""
        try:
            return True  # Service is running if we get here
        except Exception as e:
            self.logger.error(f"Error checking encoding status: {str(e)}")
            return False

    def encode_documents(self, documents: List[str]) -> EncodeResponse:
        """Encode documents and store their embeddings"""
        try:
            if not documents:
                return EncodeResponse(
                    success=False,
                    message="No documents to encode",
                    timestamp=datetime.now(),
                    documents_count=0
                )

            # Generate embeddings for each document
            embeddings = self.model.encode(documents, show_progress_bar=True)
            
            # Store documents with their embeddings
            for i, (text, embedding) in enumerate(zip(documents, embeddings)):
                doc_id = f"doc_{len(self.document_store.documents) + i}"
                self.document_store.add_document(doc_id, text, embedding)

            return EncodeResponse(
                success=True,
                message="Documents encoded and stored successfully",
                timestamp=datetime.now(),
                documents_count=len(documents)
            )

        except Exception as e:
            self.logger.error(f"Error encoding documents: {str(e)}")
            return EncodeResponse(
                success=False,
                message="Error encoding documents",
                timestamp=datetime.now(),
                documents_count=0,
                error=str(e)
            )

    def search_documents(self, query: str, top_k: int = 5) -> List[dict]:
        """Search for similar documents using the query"""
        try:
            # Encode the query
            query_embedding = self.model.encode(query)
            
            # Search for similar documents
            results = self.document_store.search(query_embedding, top_k)
            
            return results
        except Exception as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            return []

# Initialize the FastAPI application
app = FastAPI()
service = EncodeService()

# Input models
class DocumentsInput(BaseModel):
    documents: List[str]

class SearchInput(BaseModel):
    query: str
    top_k: Optional[int] = 5

# Endpoint to encode documents
@app.post("/encode_documents")
def encode_documents(input_data: DocumentsInput):
    response = service.encode_documents(input_data.documents)
    return response.__dict__

# Endpoint to search documents
@app.post("/search_documents")
def search_documents(request: SearchInput):
    results = service.search_documents(request.query, request.top_k)
    # Debug logging
    print(f"Search results: {results}")  # Add this line
    return {"results": [result.__dict__ for result in results]}

# Endpoint to check the service status
@app.get("/")
def get_status():
    return {"status": "Service is running"}

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

