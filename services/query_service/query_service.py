from fastapi import FastAPI
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Optional
import requests
import logging
import numpy as np

@dataclass
class SearchResult:
    text: str
    similarity_score: float

class QueryService:
    def __init__(self, encode_service_url: str = "http://127.0.0.1:8000"):
        self.encode_service_url = encode_service_url
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search documents based on query using embeddings"""
        try:
            # Send search request to encode_service which handles embeddings
            response = requests.post(
                f"{self.encode_service_url}/search_documents",
                json={
                    "query": query,
                    "top_k": top_k
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return [
                    SearchResult(
                        text=doc["text"],
                        similarity_score=doc["similarity_score"]
                    )
                    for doc in data.get("relevant_documents", [])
                ]
            else:
                self.logger.error(f"Search request failed: {response.text}")
                return []
        
        except Exception as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            return []

# Initialize FastAPI app
app = FastAPI()
service = QueryService()

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

@app.post("/search_documents")
def search_documents(request: SearchRequest):
    results = service.search(request.query, request.top_k)
    return {"results": [result.__dict__ for result in results]}

@app.get("/")
def get_status():
    return {"status": "Service is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
