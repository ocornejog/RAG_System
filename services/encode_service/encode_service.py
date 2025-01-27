from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
from dataclasses import dataclass
import logging
import numpy as np
import json
from pathlib import Path
import os
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the response structure
@dataclass
class EncodeResponse:
    success: bool
    message: str
    timestamp: datetime
    documents_count: int
    error: Optional[str] = None

class DocumentRequest(BaseModel):
    documents: List[str]

class SearchInput(BaseModel):
    query: str
    top_k: Optional[int] = 5

class DocumentStore:
    def __init__(self, embeddings_file=None):
        self.documents: Dict[str, dict] = {}
        self.embeddings_file = embeddings_file or os.getenv("EMBEDDINGS_FILE", "embeddings.json")
        self._ensure_data_directory_exists()
        self._load_embeddings()

    def _ensure_data_directory_exists(self):
        """Ensure the data directory exists"""
        data_dir = Path(self.embeddings_file).parent
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created data directory: {data_dir}")

    def _load_embeddings(self):
        """Load existing embeddings from file"""
        try:
            if Path(self.embeddings_file).exists():
                with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for doc in data:
                        doc['embedding'] = np.array(doc['embedding'])
                        self.documents[doc['id']] = doc
                logger.info(f"Loaded {len(data)} documents from {self.embeddings_file}")
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")

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
            
            logger.info(f"Saved {len(serializable_docs)} documents to {self.embeddings_file}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            raise

    def add_document(self, doc_id: str, text: str, embedding: np.ndarray):
        """Add a document with its embedding"""
        self.documents[doc_id] = {
            'id': doc_id,
            'text': text,
            'embedding': embedding,
            'timestamp': datetime.now().isoformat()
        }
        self._save_embeddings()
        logger.info(f"Added document with ID: {doc_id}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """Search for similar documents using cosine similarity"""
        if not self.documents:
            logger.warning("No documents available for search")
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

class EncodeService:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.document_store = DocumentStore(os.getenv("EMBEDDINGS_FILE"))
        logger.info(f"Initialized EncodeService with model: {model_name}")

    def encode_documents(self, documents: List[str]) -> EncodeResponse:
        """Encode documents using sentence-transformers"""
        try:
            if not documents:
                return EncodeResponse(
                    success=False,
                    message="No documents to encode",
                    timestamp=datetime.now(),
                    documents_count=0
                )

            # Encode documents using sentence-transformers
            embeddings = self.model.encode(documents, convert_to_numpy=True)
            
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
            logger.error(f"Error encoding documents: {str(e)}")
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
            # Encode the query using sentence-transformers
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # Search for similar documents
            results = self.document_store.search(query_embedding, top_k)
            
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

# Initialize FastAPI app and service
app = FastAPI(title="Document Encoding Service")
service = EncodeService()

@app.post("/encode_documents", response_model=Dict[str, str])
async def encode_documents(request: DocumentRequest):
    """Encode and store documents"""
    try:
        response = service.encode_documents(request.documents)
        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)
        return {
            "status": "success",
            "message": f"Successfully encoded {response.documents_count} documents"
        }
    except Exception as e:
        logger.error(f"Error in encode_documents endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_documents")
async def search_documents(request: SearchInput):
    """Search for similar documents"""
    try:
        results = service.search_documents(request.query, request.top_k)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in search_documents endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/check_storage")
async def check_storage():
    """Check if storage is properly configured"""
    try:
        # Check data directory
        data_dir = Path("/app/data")
        if not data_dir.exists():
            return {"status": "error", "message": "Data directory does not exist"}
            
        # Try to write a test file
        test_file = data_dir / "test.txt"
        test_file.write_text("test")
        test_file.unlink()  # Remove test file
        
        # Check embeddings file
        embeddings_file = Path(os.getenv("EMBEDDINGS_FILE", "/app/data/embeddings.json"))
        embeddings_exists = embeddings_file.exists()
        
        return {
            "status": "ok",
            "data_dir_exists": True,
            "data_dir_writable": True,
            "embeddings_file_exists": embeddings_exists
        }
    except Exception as e:
        logger.error(f"Error in check_storage endpoint: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)