from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import requests
import logging
import time
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import google.generativeai as genai
from google.generativeai import GenerativeModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Gemini AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=GOOGLE_API_KEY)
model = GenerativeModel('gemini-pro')

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0, le=1)
    max_relevant_chunks: int = Field(default=3, ge=1, le=5)

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]

class ChatService:
    def __init__(self, query_service_url: str = "http://query-service.rag-system:8001"):
        self.query_service_url = query_service_url
        self.session = self._setup_http_client()
        self.stats = {
            "total_chats": 0,
            "successful_chats": 0,
            "total_time": 0,
            "history": []
        }
        logger.info(f"Initialized ChatService with query service URL: {query_service_url}")

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

    async def get_relevant_documents(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Get relevant documents from query service"""
        try:
            response = self.session.post(
                f"{self.query_service_url}/search_documents",
                json={"query": query, "top_k": top_k},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return [
                {
                    'text': doc['text'],
                    'similarity': doc['similarity_score']
                }
                for doc in data.get('results', [])
            ]
        except Exception as e:
            logger.error(f"Error getting relevant documents: {str(e)}")
            raise

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process chat request with Gemini AI"""
        start_time = time.time()
        success = False
        user_message = None
        
        try:
            # Get the latest user message
            user_message = next(
                (msg.content for msg in reversed(request.messages) 
                 if msg.role == "user"),
                None
            )
            
            if not user_message:
                raise HTTPException(
                    status_code=400,
                    detail="No user message found in the conversation"
                )
            
            # Get relevant documents
            relevant_docs = await self.get_relevant_documents(
                user_message, 
                request.max_relevant_chunks
            )
            
            # Prepare context
            context = "Available context:\n\n"
            for i, doc in enumerate(relevant_docs, 1):
                context += f"{i}. {doc['text']}\n"
            
            # Prepare prompt
            prompt = f"""{context}\n
Using the provided context as a starting point, answer the following question.
You can:
- Use information from the context directly
- Make logical connections with the provided information
- Make reasonable extrapolations from the context
- Indicate when you're making assumptions or generalizations
- Specify if some aspects of the answer require more context

Question: {user_message}

Answer:"""

            # Configure Gemini AI
            generation_config = {
                "temperature": max(request.temperature, 0.7),
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 2048,
            }

            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]

            # Generate response
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            
            success = True
            duration = time.time() - start_time
            
            # Update statistics
            self.stats["total_chats"] += 1
            self.stats["successful_chats"] += 1
            self.stats["total_time"] += duration
            
            # Log conversation details
            self.stats["history"].append({
                "timestamp": datetime.now(),
                "message_length": len(user_message),
                "response_length": len(response.text),
                "num_sources": len(relevant_docs),
                "avg_similarity": sum(doc['similarity'] for doc in relevant_docs) / len(relevant_docs) if relevant_docs else 0,
                "duration": duration,
                "success": True
            })
            
            return ChatResponse(
                response=response.text,
                sources=relevant_docs
            )
        
        except Exception as e:
            success = False
            duration = time.time() - start_time
            
            # Log failed attempt
            self.stats["history"].append({
                "timestamp": datetime.now(),
                "message_length": len(user_message) if user_message else 0,
                "response_length": 0,
                "num_sources": 0,
                "avg_similarity": 0,
                "duration": duration,
                "success": False,
                "error": str(e)
            })
            
            logger.error(f"Error in chat: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize FastAPI app
app = FastAPI(title="Chat Service")
service = ChatService()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat request"""
    return await service.chat(request)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check query service
        response = requests.get(f"{service.query_service_url}/health")
        query_service_healthy = response.status_code == 200
    except:
        query_service_healthy = False

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "query_service_healthy": query_service_healthy,
        "stats": {
            "total_chats": service.stats["total_chats"],
            "successful_chats": service.stats["successful_chats"],
            "average_response_time": service.stats["total_time"] / max(service.stats["total_chats"], 1)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)