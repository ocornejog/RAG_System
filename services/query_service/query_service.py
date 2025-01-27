from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import requests
import logging
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    text: str
    similarity_score: float

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryService:
    def __init__(self, encode_service_url: str = "http://encode-service.rag-system:8000"):
        self.encode_service_url = encode_service_url
        self.session = self._setup_http_client()
        logger.info(f"Initialized QueryService with encode service URL: {encode_service_url}")

    def _setup_http_client(self):
        """Setup HTTP client with retry logic"""
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

    def check_encode_service(self) -> Dict[str, Any]:
        """Check if encode service is available"""
        try:
            response = self.session.get(f"{self.encode_service_url}/health")
            if response.status_code == 200:
                return {
                    "status": "available",
                    "details": response.json()
                }
            return {
                "status": "unavailable",
                "details": f"Status code: {response.status_code}"
            }
        except Exception as e:
            logger.error(f"Encode service health check failed: {str(e)}")
            return {
                "status": "error",
                "details": str(e)
            }

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search documents based on query using embeddings"""
        try:
            # Check encode service health
            health_status = self.check_encode_service()
            if health_status["status"] != "available":
                raise HTTPException(
                    status_code=503,
                    detail=f"Encode service unavailable: {health_status['details']}"
                )

            # Make search request
            response = self.session.post(
                f"{self.encode_service_url}/search_documents",
                json={
                    "query": query,
                    "top_k": top_k
                },
                timeout=10
            )
            
            # Handle response
            if response.status_code == 200:
                data = response.json()
                results = [
                    SearchResult(
                        text=doc["text"],
                        similarity_score=doc["similarity_score"]
                    )
                    for doc in data.get("results", [])
                ]
                logger.info(f"Found {len(results)} results for query: {query}")
                return results
            else:
                logger.error(f"Search request failed: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Search request failed: {response.text}"
                )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize FastAPI app
app = FastAPI(title="Query Service")
service = QueryService()

@app.post("/search_documents")
async def search_documents(request: SearchRequest):
    """Search for documents using the query service"""
    try:
        results = await service.search(request.query, request.top_k)
        return {
            "status": "success",
            "results": [{"text": r.text, "similarity_score": r.similarity_score} for r in results]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search_documents endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    encode_status = service.check_encode_service()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "encode_service": encode_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)