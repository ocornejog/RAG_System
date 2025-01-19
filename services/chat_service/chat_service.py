from fastapi import FastAPI
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Dict, Any
import requests
import logging

@dataclass
class ChatResponse:
    response: str
    sources: List[Dict[str, Any]]

class ChatService:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def chat(self, message: str) -> ChatResponse:
        """Send a chat message and get response"""
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "messages": [{"role": "user", "content": message}],
                    "temperature": 0.7,
                    "max_relevant_chunks": 3
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return ChatResponse(
                    response=data["response"],
                    sources=data["sources"]
                )
            return ChatResponse(
                response="Error getting response from server",
                sources=[]
            )
            
        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}")
            return ChatResponse(
                response=f"Error: {str(e)}",
                sources=[]
            )

# Initialize the FastAPI application
app = FastAPI()
service = ChatService()

# Input model for documents
class DocumentsInput(BaseModel):
    message: str

# Endpoint to encode documents
@app.post("/chat")
def chat(input_data: DocumentsInput):
    result = service.chat(input_data.message)
    return {"response": result.response, "sources": result.sources}

# Endpoint to check the service status
@app.get("/")
def get_status():
    return {"status": "Service is running"}

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
