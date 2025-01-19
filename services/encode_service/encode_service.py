from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass
import logging

# Define the response structure
@dataclass
class EncodeResponse:
    success: bool
    message: str
    timestamp: datetime
    documents_count: int
    error: Optional[str] = None

# Class for the encoding service
class EncodeService:
    def __init__(self):
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def encode_documents(self, documents: List[str]) -> EncodeResponse:
        try:
            # Simulates the processing logic
            if documents:
                return EncodeResponse(
                    success=True,
                    message="Documents encoded successfully",
                    timestamp=datetime.now(),
                    documents_count=len(documents),
                )
            else:
                return EncodeResponse(
                    success=False,
                    message="No documents to encode",
                    timestamp=datetime.now(),
                    documents_count=0,
                )
        except Exception as e:
            self.logger.error(f"Error encoding documents: {str(e)}")
            return EncodeResponse(
                success=False,
                message="Error encoding documents",
                timestamp=datetime.now(),
                documents_count=0,
                error=str(e),
            )

# Initialize the FastAPI application
app = FastAPI()
service = EncodeService()

# Input model for documents
class DocumentsInput(BaseModel):
    documents: List[str]

# Endpoint to encode documents
@app.post("/encode_documents")
def encode_documents(input_data: DocumentsInput):
    response = service.encode_documents(input_data.documents)
    return response.__dict__

# Endpoint to check the service status
@app.get("/")
def get_status():
    return {"status": "Service is running"}

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

