from fastapi import FastAPI
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Optional
import requests
import logging

@dataclass
class SearchResult:
    text: str
    similarity_score: float

class QueryService:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def search(self, query: str) -> List[SearchResult]:
        """Search documents based on query"""
        try:
            response = requests.post(
                f"{self.base_url}/search_documents",
                json={"query": query}
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
            return []
        
        except Exception as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            return []
        
# Initialize the FastAPI application
app = FastAPI()
service = QueryService()

# Input model for documents
class DocumentsInput(BaseModel):
    query: str

# Endpoint to encode documents
@app.post("/search_documents")
def search_documents(input_data: DocumentsInput):
    results = service.search(input_data.query)
    return {"results": [result.__dict__ for result in results]}

# Endpoint to check the service status
@app.get("/")
def get_status():
    return {"status": "Service is running"}

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
