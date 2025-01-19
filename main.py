import tkinter as tk
from tkinter import ttk, scrolledtext
from services.composite_service import CompositeService
from services.query_service.query_service import QueryService
from services.chat_service.chat_service import ChatService
import threading

class SimpleRAGInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG System Interface")
        
        # Initialize single service
        self.composite_service = CompositeService()
        
        # Start document monitoring in a separate thread
        threading.Thread(
            target=self.composite_service.start_document_monitoring,
            daemon=True
        ).start()
        
        self._create_widgets()

    def _create_widgets(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(pady=10, expand=True, fill="both")
        
        # Search tab
        search_frame = ttk.Frame(notebook)
        notebook.add(search_frame, text="Search")
        self._create_search_tab(search_frame)
        
        # Chat tab
        chat_frame = ttk.Frame(notebook)
        notebook.add(chat_frame, text="Chat")
        self._create_chat_tab(chat_frame)
        
        # Documents tab
        docs_frame = ttk.Frame(notebook)
        notebook.add(docs_frame, text="Documents")
        self._create_docs_tab(docs_frame)

        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")
        self._create_stats_tab(stats_frame)

    def _create_stats_tab(self, parent):
        # Système de rafraîchissement automatique
        auto_refresh_frame = ttk.Frame(parent)
        auto_refresh_frame.pack(fill="x", padx=5, pady=5)
        
        self.auto_refresh_var = tk.BooleanVar(value=False)
        auto_refresh_check = ttk.Checkbutton(
            auto_refresh_frame,
            text="Auto-refresh (5s)",
            variable=self.auto_refresh_var,
            command=self._toggle_auto_refresh
        )
        auto_refresh_check.pack(side="left")
        
        refresh_btn = ttk.Button(
            auto_refresh_frame,
            text="Refresh Now",
            command=self._refresh_statistics
        )
        refresh_btn.pack(side="right")

        # Notebook pour les différentes catégories de statistiques
        stats_notebook = ttk.Notebook(parent)
        stats_notebook.pack(expand=True, fill="both", padx=5, pady=5)

        # Onglet System
        system_frame = ttk.Frame(stats_notebook)
        stats_notebook.add(system_frame, text="System")
        
        self.system_stats_text = scrolledtext.ScrolledText(
            system_frame,
            wrap=tk.WORD,
            height=10
        )
        self.system_stats_text.pack(expand=True, fill="both")

        # Onglet Documents
        docs_stats_frame = ttk.Frame(stats_notebook)
        stats_notebook.add(docs_stats_frame, text="Documents")
        
        self.docs_stats_text = scrolledtext.ScrolledText(
            docs_stats_frame,
            wrap=tk.WORD,
            height=10
        )
        self.docs_stats_text.pack(expand=True, fill="both")

        # Onglet Search
        search_stats_frame = ttk.Frame(stats_notebook)
        stats_notebook.add(search_stats_frame, text="Search")
        
        self.search_stats_text = scrolledtext.ScrolledText(
            search_stats_frame,
            wrap=tk.WORD,
            height=10
        )
        self.search_stats_text.pack(expand=True, fill="both")

        # Onglet Chat
        chat_stats_frame = ttk.Frame(stats_notebook)
        stats_notebook.add(chat_stats_frame, text="Chat")
        
        self.chat_stats_text = scrolledtext.ScrolledText(
            chat_stats_frame,
            wrap=tk.WORD,
            height=10
        )
        self.chat_stats_text.pack(expand=True, fill="both")

        # Rafraîchissement initial
        self._refresh_statistics()

    def _toggle_auto_refresh(self):
        if self.auto_refresh_var.get():
            self._schedule_refresh()

    def _schedule_refresh(self):
        if self.auto_refresh_var.get():
            self._refresh_statistics()
            self.root.after(5000, self._schedule_refresh)

    def _refresh_statistics(self):
        try:
            # Système
            stats = self.composite_service.get_statistics()
            
            # Mise à jour System Stats
            self.system_stats_text.delete(1.0, tk.END)
            self.system_stats_text.insert(tk.END, "=== System Statistics ===\n\n")
            self.system_stats_text.insert(tk.END, f"Uptime: {stats['uptime']:.2f} seconds\n")
            self.system_stats_text.insert(tk.END, f"Memory Usage: {stats['memory_usage']:.2f} MB\n")
            self.system_stats_text.insert(tk.END, f"Total Embeddings: {stats['total_embeddings']}\n")
            self.system_stats_text.insert(tk.END, f"Embedding Dimension: {stats['embedding_dimension']}\n")
            
            # Mise à jour Document Stats
            self.docs_stats_text.delete(1.0, tk.END)
            self.docs_stats_text.insert(tk.END, "=== Document Statistics ===\n\n")
            doc_stats = stats['document_stats']
            self.docs_stats_text.insert(tk.END, f"Total Documents: {doc_stats['total_documents']}\n")
            self.docs_stats_text.insert(tk.END, f"Total Tokens: {doc_stats['total_tokens']}\n")
            self.docs_stats_text.insert(tk.END, f"Average Length: {doc_stats['average_document_length']:.2f}\n")
            self.docs_stats_text.insert(tk.END, f"Success Rate: {doc_stats['processing_success_rate']:.2f}%\n")
            self.docs_stats_text.insert(tk.END, "\nDocument Types:\n")
            for dtype, count in doc_stats['document_types'].items():
                self.docs_stats_text.insert(tk.END, f"  {dtype}: {count}\n")
            
            # Mise à jour Search Stats
            self.search_stats_text.delete(1.0, tk.END)
            self.search_stats_text.insert(tk.END, "=== Search Statistics ===\n\n")
            search_stats = stats['search_stats']
            self.search_stats_text.insert(tk.END, f"Total Searches: {search_stats['total_searches']}\n")
            self.search_stats_text.insert(tk.END, f"Average Time: {search_stats['average_search_time']:.3f}s\n")
            self.search_stats_text.insert(tk.END, f"Success Rate: {search_stats['search_success_rate']:.2f}%\n")
            self.search_stats_text.insert(tk.END, "\nTop Search Terms:\n")
            for term in search_stats['top_search_terms']:
                self.search_stats_text.insert(tk.END, f"  {term['term']}: {term['count']} times\n")
            
            # Mise à jour Chat Stats
            self.chat_stats_text.delete(1.0, tk.END)
            self.chat_stats_text.insert(tk.END, "=== Chat Statistics ===\n\n")
            chat_stats = stats['chat_stats']
            self.chat_stats_text.insert(tk.END, f"Total Chats: {chat_stats['total_chats']}\n")
            self.chat_stats_text.insert(tk.END, f"Average Response Time: {chat_stats['average_response_time']:.3f}s\n")
            self.chat_stats_text.insert(tk.END, f"Average Sources Used: {chat_stats['average_sources_used']:.2f}\n")
            self.chat_stats_text.insert(tk.END, f"Average Relevance Score: {chat_stats['average_relevance_score']:.3f}\n")
            self.chat_stats_text.insert(tk.END, f"Success Rate: {chat_stats['chat_success_rate']:.2f}%\n")

        except Exception as e:
            for text_widget in [self.system_stats_text, self.docs_stats_text, 
                              self.search_stats_text, self.chat_stats_text]:
                text_widget.delete(1.0, tk.END)
                text_widget.insert(tk.END, f"Error refreshing statistics: {str(e)}")

    def _create_search_tab(self, parent):
        # Search input
        search_frame = ttk.Frame(parent)
        search_frame.pack(pady=5, padx=5, fill="x")
        
        self.search_entry = ttk.Entry(search_frame)
        self.search_entry.pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        search_btn = ttk.Button(
            search_frame,
            text="Search",
            command=self._handle_search
        )
        search_btn.pack(side="right")
        
        # Results area
        self.search_results = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            height=20
        )
        self.search_results.pack(pady=5, padx=5, expand=True, fill="both")

    def _create_chat_tab(self, parent):
        # Chat history
        self.chat_history = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            height=15
        )
        self.chat_history.pack(pady=5, padx=5, expand=True, fill="both")
        
        # Input area
        input_frame = ttk.Frame(parent)
        input_frame.pack(pady=5, padx=5, fill="x")
        
        self.chat_entry = ttk.Entry(input_frame)
        self.chat_entry.pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        send_btn = ttk.Button(
            input_frame,
            text="Send",
            command=self._handle_chat
        )
        send_btn.pack(side="right")



    def _handle_search(self):
        query = self.search_entry.get()
        if not query:
            return
            
        self.search_results.delete(1.0, tk.END)
        results = self.composite_service.search(query)
        
        for i, result in enumerate(results, 1):
            self.search_results.insert(
                tk.END,
                f"\n{i}. Score: {result.similarity_score:.3f}\n"
                f"{result.text}\n"
                f"{'-' * 80}\n"
            )

    def _handle_chat(self):
        message = self.chat_entry.get()
        if not message:
            return
            
        # Clear input and chat history
        self.chat_entry.delete(0, tk.END)
        self.chat_history.delete(1.0, tk.END)
        
        # Add loading message
        self.chat_history.insert(tk.END, "Processing your request...\n")
        self.chat_history.update()
        
        try:
            # Get response through composite service
            response = self.composite_service.chat(message)
            
            # Clear loading message and show conversation
            self.chat_history.delete(1.0, tk.END)
            
            # Add user message
            self.chat_history.insert(tk.END, f"Question:\n{message}\n\n")
            
            # Add response
            self.chat_history.insert(tk.END, f"Réponse:\n{response.response}\n")
            
            # Filter and add relevant sources
            relevant_sources = [
                source for source in response.sources 
                if source['similarity'] * 100 > 45  # Filtrage des sources > 45%
            ]
            
            if relevant_sources:
                self.chat_history.insert(tk.END, "\nSources pertinentes:\n")
                for i, source in enumerate(relevant_sources, 1):
                    similarity = source['similarity'] * 100
                    self.chat_history.insert(
                        tk.END,
                        f"\n{i}. ({similarity:.1f}%) {source['text']}\n"
                        f"{'-' * 40}\n"
                    )
            # Optionnel : afficher le nombre de sources filtrées
            filtered_count = len(response.sources) - len(relevant_sources)
            if filtered_count > 0:
                self.chat_history.insert(
                    tk.END,
                    f"\n{filtered_count} sources de moindre pertinence ont été masquées.\n"
                )
        
        except Exception as e:
            self.chat_history.delete(1.0, tk.END)
            self.chat_history.insert(tk.END, f"Une erreur est survenue : {str(e)}\n")
        
        finally:
            self.chat_history.see(tk.END)


    def _create_docs_tab(self, parent):
        # Stats frame
        stats_frame = ttk.LabelFrame(parent, text="Processing Statistics")
        stats_frame.pack(pady=5, padx=5, fill="x")
        
        self.stats_labels = {
            "total": ttk.Label(stats_frame, text="Total: 0"),
            "successful": ttk.Label(stats_frame, text="Successful: 0"),
            "failed": ttk.Label(stats_frame, text="Failed: 0"),
            "rate": ttk.Label(stats_frame, text="Success Rate: 0%")
        }
        
        for label in self.stats_labels.values():
            label.pack(side="left", padx=10, pady=5)
        
        # Documents list
        self.docs_text = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            height=20
        )
        self.docs_text.pack(pady=5, padx=5, expand=True, fill="both")
        
        refresh_btn = ttk.Button(
            parent,
            text="Refresh",
            command=self._refresh_docs
        )
        refresh_btn.pack(pady=5)
        
        # Initial refresh
        self._refresh_docs()

    def _refresh_docs(self):
        self.docs_text.delete(1.0, tk.END)
        docs = self.composite_service.get_processed_documents()
        
        # Update statistics
        stats = self.composite_service.get_processing_statistics()
        self.stats_labels["total"].config(text=f"Total: {stats['total_documents']}")
        self.stats_labels["successful"].config(text=f"Successful: {stats['successful_encodings']}")
        self.stats_labels["failed"].config(text=f"Failed: {stats['failed_encodings']}")
        self.stats_labels["rate"].config(text=f"Success Rate: {stats['success_rate']:.1f}%")
        
        # Update document list
        for filepath, doc in docs.items():
            status = "✓" if doc.encode_response.success else "✗"
            error_info = f"\nError: {doc.encode_response.error}" if doc.encode_response.error else ""
            
            self.docs_text.insert(
                tk.END,
                f"\n{status} {filepath}\n"
                f"Processed at: {doc.processed_time}\n"
                f"Status: {doc.encode_response.message}{error_info}\n"
                f"{'-' * 80}\n"
            )
    
    

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleRAGInterface(root)
    root.mainloop()