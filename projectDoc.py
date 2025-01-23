from google.generativeai import GenerativeModel
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from datetime import datetime
import logging
from dotenv import load_dotenv
import time
from collections import deque
from pathlib import Path


# Config. Google GEMINI
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", "embeddings.json")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "paraphrase-MiniLM-L6-v2")

genai.configure(api_key=GOOGLE_API_KEY)
model = GenerativeModel('gemini-pro')
start_time = datetime.now()

document_history = []
search_history = []
search_stats = {
    "history": deque(maxlen=1000),  # Limite l'historique à 1000 entrées
    "total_searches": 0,
    "total_time": 0,
    "successful_searches": 0
}

# Historique et statistiques des chats
chat_history = []
chat_stats = {
    "history": deque(maxlen=1000),
    "total_chats": 0,
    "total_time": 0,
    "successful_chats": 0,
    "total_tokens": 0
}

# Initialisation FastAPI
app = FastAPI(
    title="Système RAG API",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the sentence transformer model
embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèle de données
class DocumentStats(BaseModel):
    total_documents: int
    total_tokens: int
    average_document_length: float
    document_types: Dict[str, int]
    processing_success_rate: float
    last_document_added: Optional[datetime]

class SearchStats(BaseModel):
    total_searches: int
    average_search_time: float
    top_search_terms: List[Dict[str, Any]]
    average_results_returned: float
    search_success_rate: float

class ChatStats(BaseModel):
    total_chats: int
    average_response_time: float
    average_sources_used: float
    average_relevance_score: float
    chat_success_rate: float

class SystemStats(BaseModel):
    uptime: float
    total_embeddings: int
    embedding_dimension: int
    memory_usage: float
    document_stats: DocumentStats
    search_stats: SearchStats
    chat_stats: ChatStats


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

@app.post("/log/search")
async def log_search(log: SearchLog):
    """Log search statistics"""
    search_history.append({
        "timestamp": datetime.now(),
        "query": log.query,
        "success": log.success,
        "num_results": log.num_results,
        "error": log.error
    })
    return {"status": "success"}

@app.post("/log/chat")
async def log_chat(log: ChatLog):
    """Log chat statistics"""
    chat_history.append({
        "timestamp": datetime.now(),
        "message": log.message,
        "success": log.success,
        "duration": log.duration,
        "num_sources": log.num_sources,
        "error": log.error
    })
    return {"status": "success"}


# Fonction de similarité entre deux vecteurs (embeds)
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


# Enregistre embeddings format JSON
def save_embeddings(new_documents_data: List[Dict[str, Any]]):
    """
    Sauvegarde les embeddings en préservant les documents existants
    """
    # Charger les documents existants
    try:
        with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []
    
    # Convertir les embeddings existants en dictionnaire pour recherche rapide
    existing_docs = {doc['text']: doc for doc in existing_data}
    
    # Préparer les données à sauvegarder
    serializable_data = []
    
    # Ajouter les documents existants
    for doc in existing_data:
        doc_copy = doc.copy()
        if isinstance(doc_copy['embedding'], list):
            serializable_data.append(doc_copy)
        else:
            doc_copy['embedding'] = doc_copy['embedding'].tolist()
            serializable_data.append(doc_copy)
    
    # Ajouter les nouveaux documents (avec gestion des doublons)
    next_id = max([doc['id'] for doc in serializable_data], default=-1) + 1
    for doc in new_documents_data:
        if doc['text'] not in existing_docs:  # Éviter les doublons
            doc_copy = doc.copy()
            doc_copy['id'] = next_id
            doc_copy['embedding'] = doc['embedding'].tolist()
            serializable_data.append(doc_copy)
            next_id += 1
    
    # Sauvegarder tous les documents
    with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)

# Load embeddings
def load_embeddings() -> List[Dict[str, Any]]:
    try:
        with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for doc in data:
                doc['embedding'] = np.array(doc['embedding'])
            return data
    except FileNotFoundError:
        return []

@app.get("/stats/documents", response_model=DocumentStats)
async def get_document_statistics():
    """
    Obtenir les statistiques détaillées sur les documents encodés
    """
    try:
        documents_data = load_embeddings()
        
        if not documents_data:
            return DocumentStats(
                total_documents=0,
                total_tokens=0,
                average_document_length=0,
                document_types={},
                processing_success_rate=0,
                last_document_added=None
            )

        # Calcul des statistiques
        total_docs = len(documents_data)
        doc_lengths = [len(doc['text'].split()) for doc in documents_data]
        doc_types = {}
        for doc in documents_data:
            ext = Path(doc.get('filepath', '')).suffix
            doc_types[ext] = doc_types.get(ext, 0) + 1

        return DocumentStats(
            total_documents=total_docs,
            total_tokens=sum(doc_lengths),
            average_document_length=np.mean(doc_lengths),
            document_types=doc_types,
            processing_success_rate=(len([d for d in document_history if d.get('success')]) / max(len(document_history), 1)) * 100,
            last_document_added=document_history[-1]['timestamp'] if document_history else None
        )
    except Exception as e:
        logger.error(f"Error getting document statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/search", response_model=SearchStats)
async def get_search_statistics():
    if not search_history:
        return SearchStats(
            total_searches=0,
            average_search_time=0,
            top_search_terms=[],
            average_results_returned=0,
            search_success_rate=0
        )

    total_searches = len(search_history)
    successful_searches = sum(1 for s in search_history if s.get('success', False))
    
    # Agrégation des termes de recherche
    search_terms = {}
    for search in search_history:
        term = search['query']
        search_terms[term] = search_terms.get(term, 0) + 1
    
    # Top 10 des termes les plus recherchés
    top_terms = sorted(
        [{"term": k, "count": v} for k, v in search_terms.items()],
        key=lambda x: x["count"],
        reverse=True
    )[:10]

    return SearchStats(
        total_searches=total_searches,
        average_search_time=0.0,  # À implémenter si vous suivez le temps
        top_search_terms=top_terms,
        average_results_returned=sum(s.get('num_results', 0) for s in search_history) / total_searches,
        search_success_rate=(successful_searches / total_searches * 100) if total_searches > 0 else 0
    )

# Modifier get_chat_statistics pour inclure plus d'informations
@app.get("/stats/chat", response_model=ChatStats)
async def get_chat_statistics():
    if not chat_history:
        return ChatStats(
            total_chats=0,
            average_response_time=0,
            average_sources_used=0,
            average_relevance_score=0,
            chat_success_rate=0
        )

    total_chats = len(chat_history)
    successful_chats = sum(1 for c in chat_history if c.get('success', False))
    
    return ChatStats(
        total_chats=total_chats,
        average_response_time=sum(c.get('duration', 0) for c in chat_history) / total_chats,
        average_sources_used=sum(c.get('num_sources', 0) for c in chat_history) / total_chats,
        average_relevance_score=0.0,  # À implémenter si vous suivez les scores
        chat_success_rate=(successful_chats / total_chats * 100) if total_chats > 0 else 0
    )
@app.get("/stats", response_model=SystemStats)
async def get_system_statistics():
    """
    Obtenir toutes les statistiques du système
    """
    try:
        import psutil
        
        # Obtenir les statistiques des autres endpoints
        doc_stats = await get_document_statistics()
        search_stats = await get_search_statistics()
        chat_stats = await get_chat_statistics()
        
        # Calculer les statistiques système
        process = psutil.Process()
        uptime = (datetime.now() - start_time).total_seconds()
        
        documents_data = load_embeddings()
        embedding_dim = len(documents_data[0]['embedding']) if documents_data else 0
        
        return SystemStats(
            uptime=uptime,
            total_embeddings=len(documents_data) if documents_data else 0,
            embedding_dimension=embedding_dim,
            memory_usage=process.memory_info().rss / 1024 / 1024,  # En MB
            document_stats=doc_stats,
            search_stats=search_stats,
            chat_stats=chat_stats
        )
    except Exception as e:
        logger.error(f"Error getting system statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode_documents", response_model=Dict[str, str])
async def encode_documents(request: DocumentRequest):
    """
    Encode documents into embeddings and store them.
    """
    try:
        # Generate embeddings for each document
        new_documents_data = []
        for text in request.documents:
            embedding = embedding_model.encode(text)
            timestamp = datetime.now().isoformat()
            new_documents_data.append({
                'id': -1,  # Will be replaced during save
                'text': text,
                'embedding': embedding,
                'timestamp': timestamp,
                'status': 'active'
            })
        
        # Save embeddings to file (merge with existing)
        save_embeddings(new_documents_data)
        
        return {
            "status": "success",
            "message": f"Successfully encoded and added {len(request.documents)} documents"
        }
    
    except Exception as e:
        logger.error(f"Error encoding documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_documents", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for relevant documents based on a query.
    """
    start_time = time.time()
    success = False
    
    try:
        # Load stored embeddings
        documents_data = load_embeddings()
        if not documents_data:
            raise HTTPException(status_code=404, detail="No documents have been encoded yet")
        
        # Encode the query
        query_embedding = embedding_model.encode(request.query)
        
        # Calculate similarities and sort documents
        results = []
        for doc in documents_data:
            similarity = cosine_similarity(query_embedding, doc['embedding'])
            results.append(SearchResult(
                id=doc['id'],
                text=doc['text'],
                similarity_score=float(similarity)
            ))
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        success = True
        
        # Mettre à jour les statistiques
        search_stats["total_searches"] += 1
        if success:
            search_stats["successful_searches"] += 1
        
        end_time = time.time()
        duration = end_time - start_time
        search_stats["total_time"] += duration
        
        # Enregistrer les détails de la recherche
        search_stats["history"].append({
            "timestamp": datetime.now(),
            "query": request.query,
            "num_results": len(results[:5]),
            "duration": duration,
            "success": success,
            "top_similarity": results[0].similarity_score if results else 0
        })
        
        # Return top 5 most relevant documents
        return SearchResponse(relevant_documents=results[:5])
    
    except Exception as e:
        success = False
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Mettre à jour les statistiques même en cas d'erreur
        if not success:
            search_stats["history"].append({
                "timestamp": datetime.now(),
                "query": request.query,
                "num_results": 0,
                "duration": time.time() - start_time,
                "success": False,
                "error": str(e) if 'e' in locals() else "Unknown error"
            })

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    success = False
    
    try:
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
        
        # Recherche des documents pertinents
        search_results = await search_documents(SearchRequest(query=user_message))
        relevant_docs = [
            {
                'text': doc.text,
                'similarity': doc.similarity_score
            }
            for doc in search_results.relevant_documents[:request.max_relevant_chunks]
        ]
        
        context = "Contexte disponible :\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"{i}. {doc['text']}\n"
        
        prompt = f"""{context}\n
En utilisant le contexte fourni comme point de départ, réponds à la question suivante.
Tu peux :
- Utiliser les informations du contexte directement
- Faire des connexions logiques avec les informations fournies
- Extrapoler raisonnablement à partir du contexte
- Indiquer quand tu fais des suppositions ou des généralisations
- Préciser si certains aspects de la réponse nécessitent plus de contexte

Question: {user_message}

Réponse:"""

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

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        
        success = True
        end_time = time.time()
        duration = end_time - start_time
        
        # Mettre à jour les statistiques
        chat_stats["total_chats"] += 1
        if success:
            chat_stats["successful_chats"] += 1
        chat_stats["total_time"] += duration
        
        # Enregistrer les détails de la conversation
        chat_stats["history"].append({
            "timestamp": datetime.now(),
            "message_length": len(user_message),
            "response_length": len(response.text),
            "num_sources": len(relevant_docs),
            "avg_similarity": sum(doc['similarity'] for doc in relevant_docs) / len(relevant_docs) if relevant_docs else 0,
            "duration": duration,
            "success": success
        })
        
        return ChatResponse(
            response=response.text,
            sources=relevant_docs
        )
    
    except Exception as e:
        success = False
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if not success:
            chat_stats["history"].append({
                "timestamp": datetime.now(),
                "message_length": len(user_message) if 'user_message' in locals() else 0,
                "response_length": 0,
                "num_sources": 0,
                "avg_similarity": 0,
                "duration": time.time() - start_time,
                "success": False,
                "error": str(e) if 'e' in locals() else "Unknown error"
            })


@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "RAG System API is running (Powered by Google Gemini)",
        "endpoints": {
            "/encode_documents": "POST - Encode and store documents",
            "/search_documents": "POST - Search for relevant documents",
            "/chat": "POST - Chat with the RAG system using Gemini"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)