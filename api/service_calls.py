from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import requests
import logging
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
COMPOSITE_SERVICE_URL = os.getenv("COMPOSITE_SERVICE_URL", "http://composite-service.rag-system:8003")

# Initialize FastAPI
app = FastAPI(
    title="RAG System API Gateway",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Models
class DocumentRequest(BaseModel):
    documents: List[str]

class SearchRequest(BaseModel):
    query: str

class SearchResult(BaseModel):
    id: int
    text: str
    similarity_score: float

class SearchResponse(BaseModel):
    relevant_documents: List[SearchResult]

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0, le=1)
    max_relevant_chunks: int = Field(default=3, ge=1, le=5)

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]

# Helper function for making HTTP requests
def make_request(method: str, endpoint: str, data: dict = None) -> dict:
    """Make HTTP request to composite service with error handling"""
    try:
        url = f"{COMPOSITE_SERVICE_URL}{endpoint}"
        response = requests.request(
            method=method,
            url=url,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request to composite service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Error communicating with composite service: {str(e)}"
        )

@app.post("/encode_documents", response_model=Dict[str, str])
async def encode_documents(request: DocumentRequest):
    """Forward document encoding request to composite service"""
    try:
        return make_request("POST", "/encode_documents", {"documents": request.documents})
    except Exception as e:
        logger.error(f"Error encoding documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_documents", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Forward search request to composite service"""
    try:
        result = make_request("POST", "/search_documents", {"query": request.query})
        return SearchResponse(**result)
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Forward chat request to composite service"""
    try:
        result = make_request("POST", "/chat", request.dict())
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    """Get system statistics from composite service"""
    try:
        return make_request("GET", "/stats")
    except Exception as e:
        logger.error(f"Error getting system statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "RAG System API Gateway",
        "status": "Running",
        "endpoints": {
            "/encode_documents": "POST - Encode and store documents",
            "/search_documents": "POST - Search for relevant documents",
            "/chat": "POST - Chat with the RAG system",
            "/stats": "GET - Get system statistics"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)