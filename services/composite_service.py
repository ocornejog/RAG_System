import time
from pathlib import Path
from typing import List, Dict, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from dataclasses import dataclass
from datetime import datetime
import requests
from .encode_service.encode_service import EncodeService, EncodeResponse
import psutil

@dataclass
class DocumentResponse:
    text: str
    processed_time: datetime
    encode_response: EncodeResponse
    filepath: str

@dataclass
class SearchResult:
    text: str
    similarity_score: float

@dataclass
class ChatResponse:
    response: str
    sources: List[Dict[str, Any]]

class CompositeService:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.processed_documents: Dict[str, DocumentResponse] = {}
        self.observer = None
        self.handler = None
        self.encode_service = EncodeService(base_url)
        self._setup_logging()
        self.start_time = datetime.now()

    def get_statistics(self) -> dict:
        """
        Récupère toutes les statistiques du système via les appels API.
        """
        try:
            # Statistiques système
            process = psutil.Process()
            system_uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Récupération des stats via les différents endpoints
            doc_stats_response = requests.get(f"{self.base_url}/stats/documents")
            search_stats_response = requests.get(f"{self.base_url}/stats/search")
            chat_stats_response = requests.get(f"{self.base_url}/stats/chat")
            
            if not all(r.status_code == 200 for r in [doc_stats_response, search_stats_response, chat_stats_response]):
                raise Exception("Failed to get complete statistics from API")
            
            doc_stats = doc_stats_response.json()
            search_stats = search_stats_response.json()
            chat_stats = chat_stats_response.json()
            
            return {
                'uptime': system_uptime,
                'memory_usage': process.memory_info().rss / 1024 / 1024,
                'total_embeddings': len(self.processed_documents),
                'embedding_dimension': 384,  # paraphrase-MiniLM-L6-v2 dimension
                
                'document_stats': doc_stats,
                'search_stats': search_stats,
                'chat_stats': chat_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return self._get_default_statistics()

    def _calculate_average_doc_length(self) -> float:
        """Calcule la longueur moyenne des documents"""
        if not self.processed_documents:
            return 0.0
        lengths = [len(doc.text.split()) for doc in self.processed_documents.values()]
        return sum(lengths) / len(lengths)

    def _get_document_types(self) -> dict:
        """Récupère les types de documents et leur compte"""
        type_counts = {}
        for filepath in self.processed_documents:
            ext = Path(filepath).suffix
            type_counts[ext] = type_counts.get(ext, 0) + 1
        return type_counts

    def _calculate_success_rate(self) -> float:
        """Calcule le taux de succès du traitement des documents"""
        if not self.processed_documents:
            return 0.0
        successful = sum(1 for doc in self.processed_documents.values() 
                        if doc.encode_response.success)
        return (successful / len(self.processed_documents)) * 100

    def _get_default_statistics(self) -> dict:
        """Retourne des statistiques par défaut en cas d'erreur"""
        return {
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'memory_usage': 0,
            'total_embeddings': 0,
            'embedding_dimension': 0,
            'document_stats': {
                'total_documents': 0,
                'total_tokens': 0,
                'average_document_length': 0,
                'document_types': {},
                'processing_success_rate': 0
            },
            'search_stats': {
                'total_searches': 0,
                'average_search_time': 0,
                'top_search_terms': [],
                'search_success_rate': 0
            },
            'chat_stats': {
                'total_chats': 0,
                'average_response_time': 0,
                'average_sources_used': 0,
                'average_relevance_score': 0,
                'chat_success_rate': 0
            }
        }

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def start_document_monitoring(self, docs_dir: str = "documents"):
        """Start monitoring the documents directory"""
        try:
            # Check if encoding service is available
            if not self.encode_service.get_encoding_status():
                self.logger.error("Encoding service is not available")
                raise ConnectionError("Cannot connect to encoding service")
                
            path = Path(docs_dir)
            path.mkdir(exist_ok=True)
            
            self.handler = DocumentHandler(self)
            self.observer = Observer()
            self.observer.schedule(self.handler, str(path), recursive=False)
            self.observer.start()
            
            self.logger.info(f"Started monitoring directory: {path.absolute()}")
            
            # Process existing documents
            self._process_existing_documents(path)
        except Exception as e:
            self.logger.error(f"Error starting document monitoring: {str(e)}")
            raise

    def search(self, query: str) -> List[SearchResult]:
        """Search documents through the API"""
        try:
            response = requests.post(
                f"{self.base_url}/search_documents",
                json={"query": query}
            )
            
            if response.status_code == 200:
                data = response.json()
                # Log de la recherche réussie pour les stats
                requests.post(f"{self.base_url}/log/search", json={
                    "query": query,
                    "success": True,
                    "num_results": len(data["relevant_documents"])
                })
                return [
                    SearchResult(
                        text=doc["text"],
                        similarity_score=doc["similarity_score"]
                    )
                    for doc in data["relevant_documents"]
                ]
            return []
        except Exception as e:
            # Log de la recherche échouée
            requests.post(f"{self.base_url}/log/search", json={
                "query": query,
                "success": False,
                "error": str(e)
            })
            self.logger.error(f"Error searching documents: {str(e)}")
            return []

    def chat(self, message: str) -> ChatResponse:
        """Handle chat interactions"""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "messages": [{"role": "user", "content": message}],
                    "temperature": 0.7,
                    "max_relevant_chunks": 3
                }
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                # Log du chat réussi pour les stats
                requests.post(f"{self.base_url}/log/chat", json={
                    "message": message,
                    "success": True,
                    "duration": duration,
                    "num_sources": len(data["sources"])
                })
                return ChatResponse(
                    response=data["response"],
                    sources=data["sources"]
                )
            return ChatResponse(
                response="Error getting response from server",
                sources=[]
            )
        except Exception as e:
            # Log du chat échoué
            requests.post(f"{self.base_url}/log/chat", json={
                "message": message,
                "success": False,
                "error": str(e)
            })
            self.logger.error(f"Error in chat: {str(e)}")
            return ChatResponse(
                response=f"Error: {str(e)}",
                sources=[]
            )

    def process_document(self, filepath: str) -> bool:
        """Process a single document with improved error handling and retry logic"""
        try:
            if filepath in self.processed_documents:
                self.logger.info(f"Document already processed: {filepath}")
                return True

            # Add a small delay to ensure file is completely written
            time.sleep(0.5)

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                self.logger.warning(f"Empty document: {filepath}")
                return False

            # Use encode service with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    encode_response = self.encode_service.encode_documents([content])
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)  # Wait before retry
            
            # Store response regardless of success
            self.processed_documents[filepath] = DocumentResponse(
                text=content,
                processed_time=datetime.now(),
                encode_response=encode_response,
                filepath=filepath
            )
            
            if encode_response.success:
                self.logger.info(f"Successfully processed: {filepath}")
            else:
                self.logger.error(f"Failed to process {filepath}: {encode_response.error}")
                
            return encode_response.success

        except Exception as e:
            self.logger.error(f"Error processing {filepath}: {str(e)}")
            # Store error response
            error_response = EncodeResponse(
                success=False,
                message="Error processing document",
                timestamp=datetime.now(),
                documents_count=1,
                error=str(e)
            )
            self.processed_documents[filepath] = DocumentResponse(
                text="",
                processed_time=datetime.now(),
                encode_response=error_response,
                filepath=filepath
            )
            return False

    def _process_existing_documents(self, path: Path):
        """Process existing documents with improved handling"""
        try:
            files = list(path.glob("*.txt"))
            self.logger.info(f"Found {len(files)} existing documents")
            
            for file_path in files:
                try:
                    success = self.process_document(str(file_path))
                    if not success:
                        self.logger.warning(f"Failed to process existing document: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {str(e)}")
                    continue
                
        except Exception as e:
            self.logger.error(f"Error processing existing documents: {str(e)}")

    def get_processed_documents(self) -> Dict[str, DocumentResponse]:
        """Get all processed documents with their status"""
        return self.processed_documents
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        total = len(self.processed_documents)
        successful = sum(1 for doc in self.processed_documents.values() 
                        if doc.encode_response.success)
        failed = total - successful
        
        return {
            "total_documents": total,
            "successful_encodings": successful,
            "failed_encodings": failed,
            "success_rate": (successful/total * 100) if total > 0 else 0
        }

class DocumentHandler(FileSystemEventHandler):
    def __init__(self, service: CompositeService):
        self.service = service
        self._processing_files = set()

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        self._handle_file_event(event.src_path)

    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        self._handle_file_event(event.src_path)

    def _handle_file_event(self, filepath):
        """Handle file events with duplicate prevention"""
        if filepath in self._processing_files:
            return
        
        self._processing_files.add(filepath)
        try:
            time.sleep(1)  # Wait for file to be completely written
            self.service.process_document(filepath)
        finally:
            self._processing_files.remove(filepath)