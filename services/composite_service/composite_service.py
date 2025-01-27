from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import requests
import logging
from pathlib import Path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import psutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import deque

# Pydantic models for request/response
class DocumentRequest(BaseModel):
    documents: List[str]

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0, le=1)
    max_relevant_chunks: int = Field(default=3, ge=1, le=5)

class SearchLog(BaseModel):
    query: str
    success: bool
    num_results: Optional[int] = 0
    error: Optional[str] = None

class ChatLog(BaseModel):
    message: str
    success: bool
    duration: float
    num_sources: Optional[int] = 0
    error: Optional[str] = None

# Response models
class EncodeResponse:
    def __init__(self, success: bool, message: str, timestamp: datetime, documents_count: int, error: Optional[str] = None):
        self.success = success
        self.message = message
        self.timestamp = timestamp
        self.documents_count = documents_count
        self.error = error

class DocumentResponse:
    def __init__(self, text: str, processed_time: datetime, encode_response: dict, filepath: str):
        self.text = text
        self.processed_time = processed_time
        self.encode_response = encode_response
        self.filepath = filepath

class SearchResult:
    def __init__(self, text: str, similarity_score: float):
        self.text = text
        self.similarity_score = similarity_score

class ChatResponse:
    def __init__(self, message: str, response: str, timestamp: datetime):
        self.message = message
        self.response = response
        self.timestamp = timestamp

class DocumentHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        self.logger.info(f"New document detected: {event.src_path}")
        self.service.process_document(event.src_path)

    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        self.logger.info(f"Document modified: {event.src_path}")
        self.service.process_document(event.src_path)

class CompositeService:
    def __init__(self, 
                 encode_service_url: str = "http://encode-service.rag-system:8000",
                 query_service_url: str = "http://query-service.rag-system:8001",
                 chat_service_url: str = "http://chat-service.rag-system:8002",
                 docs_dir: str = "/app/documents"):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        self.encode_service_url = encode_service_url
        self.query_service_url = query_service_url
        self.chat_service_url = chat_service_url
        self.docs_dir = docs_dir
        
        # Initialize statistics tracking
        self.start_time = datetime.now()
        self.processed_documents: Dict[str, DocumentResponse] = {}
        self.search_stats = {
            "history": deque(maxlen=1000),
            "total_searches": 0,
            "total_time": 0,
            "successful_searches": 0
        }
        self.chat_stats = {
            "history": deque(maxlen=1000),
            "total_chats": 0,
            "total_time": 0,
            "successful_chats": 0
        }
        
        # Setup monitoring
        self.observer = None
        self.handler = None
        
        # Setup HTTP session
        self.session = self._setup_http_session()

    def _setup_http_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def encode_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Encode documents using the encode service"""
        try:
            response = self.session.post(
                f"{self.encode_service_url}/encode_documents",
                json={"documents": documents},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error encoding documents: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def search_documents(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search documents using the query service"""
        start_time = time.time()
        try:
            response = self.session.post(
                f"{self.query_service_url}/search_documents",
                json={"query": query, "top_k": top_k},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Update search statistics
            duration = time.time() - start_time
            self.search_stats["total_searches"] += 1
            self.search_stats["total_time"] += duration
            self.search_stats["successful_searches"] += 1
            self.search_stats["history"].append({
                "timestamp": datetime.now(),
                "query": query,
                "num_results": len(data.get("results", [])),
                "duration": duration,
                "success": True
            })
            
            return [
                SearchResult(text=doc["text"], similarity_score=doc["similarity_score"])
                for doc in data.get("results", [])
            ]
        except Exception as e:
            # Log failed search
            self.search_stats["history"].append({
                "timestamp": datetime.now(),
                "query": query,
                "num_results": 0,
                "duration": time.time() - start_time,
                "success": False,
                "error": str(e)
            })
            self.logger.error(f"Error searching documents: {str(e)}")
            return []

    async def chat(self, request: ChatRequest) -> Dict[str, Any]:
        """Handle chat interactions using the chat service"""
        start_time = time.time()
        try:
            response = self.session.post(
                f"{self.chat_service_url}/chat",
                json=request.dict(),
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Update chat statistics
            duration = time.time() - start_time
            self.chat_stats["total_chats"] += 1
            self.chat_stats["total_time"] += duration
            self.chat_stats["successful_chats"] += 1
            self.chat_stats["history"].append({
                "timestamp": datetime.now(),
                "message": request.messages[-1].content if request.messages else "",
                "duration": duration,
                "num_sources": len(data.get("sources", [])),
                "success": True
            })
            
            return data
        except Exception as e:
            # Log failed chat
            self.chat_stats["history"].append({
                "timestamp": datetime.now(),
                "message": request.messages[-1].content if request.messages else "",
                "duration": time.time() - start_time,
                "success": False,
                "error": str(e)
            })
            self.logger.error(f"Error in chat: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            process = psutil.Process()
            
            # Calculate statistics
            total_docs = len(self.processed_documents)
            successful_docs = sum(1 for doc in self.processed_documents.values() 
                                if doc.encode_response.get("success", False))
            
            search_success_rate = (self.search_stats["successful_searches"] / 
                                 max(self.search_stats["total_searches"], 1)) * 100
            
            chat_success_rate = (self.chat_stats["successful_chats"] / 
                               max(self.chat_stats["total_chats"], 1)) * 100
            
            return {
                "uptime": (datetime.now() - self.start_time).total_seconds(),
                "memory_usage": process.memory_info().rss / 1024 / 1024,
                "total_embeddings": total_docs,
                "embedding_dimension": 384,  # paraphrase-MiniLM-L6-v2 dimension
                
                "document_stats": {
                    "total_documents": total_docs,
                    "successful_encodings": successful_docs,
                    "processing_success_rate": (successful_docs / max(total_docs, 1)) * 100,
                    "document_types": self._get_document_types()
                },
                
                "search_stats": {
                    "total_searches": self.search_stats["total_searches"],
                    "average_search_time": (self.search_stats["total_time"] / 
                                          max(self.search_stats["total_searches"], 1)),
                    "search_success_rate": search_success_rate
                },
                
                "chat_stats": {
                    "total_chats": self.chat_stats["total_chats"],
                    "average_response_time": (self.chat_stats["total_time"] / 
                                            max(self.chat_stats["total_chats"], 1)),
                    "chat_success_rate": chat_success_rate
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return self._get_default_statistics()

# Initialize FastAPI app
app = FastAPI()
service = CompositeService()

@app.post("/encode_documents")
async def encode_documents(request: DocumentRequest):
    """Endpoint to encode documents"""
    return service.encode_documents(request.documents)

@app.post("/search_documents")
async def search_documents(request: SearchRequest):
    """Endpoint to search documents"""
    results = service.search_documents(request.query, request.top_k)
    return {"results": [{"text": r.text, "similarity_score": r.similarity_score} for r in results]}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Endpoint for chat interactions"""
    return await service.chat(request)

@app.get("/stats")
async def get_stats():
    """Endpoint to get system statistics"""
    return service.get_statistics()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)