import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import requests
import os
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleRAGInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG System Interface")
        
        # Default to localhost if running locally
        self.api_url = os.getenv("API_GATEWAY_URL", "http://localhost:8000")
        logger.info(f"Connecting to API Gateway at: {self.api_url}")
        
        # Initialize UI
        self._check_service()
        self._create_widgets()

    def _check_service(self):
        """Check if API gateway is available"""
        try:
            response = requests.get(f"{self.api_url}/")
            if response.status_code != 200:
                logger.error("API Gateway is not available")
                messagebox.showerror("Service Error", 
                    "API Gateway is not available. Please make sure the services are running.")
        except Exception as e:
            logger.error(f"Error connecting to API Gateway: {str(e)}")
            messagebox.showerror("Connection Error", 
                f"Cannot connect to API Gateway at {self.api_url}. Please check if services are running.")

    def _create_widgets(self):
        # Configure main window
        self.root.geometry("800x600")  # Set default window size
        
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
        
        # Statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")
        self._create_stats_tab(stats_frame)

        # Add status bar
        self.status_bar = ttk.Label(self.root, text="Connected to API Gateway", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_chat_tab(self, parent):
        # Chat history
        self.chat_history = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            height=15
        )
        self.chat_history.pack(pady=5, padx=5, expand=True, fill="both")
        
        # Input frame
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
        
        # Bind Enter key
        self.chat_entry.bind("<Return>", lambda e: self._handle_chat())

    def _create_search_tab(self, parent):
        # Search frame
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

    def _create_stats_tab(self, parent):
        # Auto-refresh controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        self.auto_refresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            control_frame,
            text="Auto-refresh (5s)",
            variable=self.auto_refresh_var,
            command=self._toggle_auto_refresh
        ).pack(side="left")
        
        ttk.Button(
            control_frame,
            text="Refresh Now",
            command=self._refresh_statistics
        ).pack(side="right")

        # Stats notebook
        stats_notebook = ttk.Notebook(parent)
        stats_notebook.pack(expand=True, fill="both", padx=5, pady=5)

        # Create sections
        self.stats_texts = {}
        for section in ["System", "Documents", "Search", "Chat"]:
            frame = ttk.Frame(stats_notebook)
            stats_notebook.add(frame, text=section)
            
            text_widget = scrolledtext.ScrolledText(
                frame,
                wrap=tk.WORD,
                height=15
            )
            text_widget.pack(expand=True, fill="both")
            self.stats_texts[section.lower()] = text_widget

        self._refresh_statistics()

    def _handle_search(self):
        query = self.search_entry.get()
        if not query:
            return
            
        self.search_results.delete(1.0, tk.END)
        try:
            response = self._make_request("POST", "/search_documents", {
                "query": query
            })
            
            if response:
                results = response.get("relevant_documents", [])
                for i, result in enumerate(results, 1):
                    self.search_results.insert(
                        tk.END,
                        f"\n{i}. Score: {result['similarity_score']:.3f}\n"
                        f"{result['text']}\n"
                        f"{'-' * 80}\n"
                    )
        except Exception as e:
            self.search_results.insert(tk.END, f"Error: {str(e)}\n")
            self.status_bar.config(text=f"Error: {str(e)}")

    def _handle_chat(self):
        message = self.chat_entry.get()
        if not message:
            return
            
        self.chat_entry.delete(0, tk.END)
        self.chat_history.insert(tk.END, f"\nYou: {message}\n")
        
        try:
            response = self._make_request("POST", "/chat", {
                "messages": [{"role": "user", "content": message}],
                "temperature": 0.7,
                "max_relevant_chunks": 3
            })
            
            if response:
                self.chat_history.insert(tk.END, f"\nAssistant: {response['response']}\n")
                if response.get('sources'):
                    self.chat_history.insert(tk.END, "\nSources:\n")
                    for i, source in enumerate(response['sources'], 1):
                        self.chat_history.insert(
                            tk.END, 
                            f"{i}. Similarity: {source['similarity']:.3f}\n"
                            f"   {source['text']}\n"
                        )
                self.chat_history.insert(tk.END, f"{'-' * 80}\n")
                self.chat_history.see(tk.END)
                self.status_bar.config(text="Message sent successfully")
        except Exception as e:
            self.chat_history.insert(tk.END, f"Error: {str(e)}\n")
            self.chat_history.see(tk.END)
            self.status_bar.config(text=f"Error: {str(e)}")

    def _make_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make request to API gateway with retry logic"""
        try:
            url = f"{self.api_url}{endpoint}"
            response = requests.request(method=method, url=url, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error making request to {endpoint}: {str(e)}")
            raise

    def _refresh_statistics(self):
        try:
            stats = self._make_request("GET", "/stats")
            if not stats:
                return
                
            # Update System Stats
            system_text = self.stats_texts['system']
            system_text.delete(1.0, tk.END)
            system_text.insert(tk.END, "=== System Statistics ===\n\n")
            system_text.insert(tk.END, f"Uptime: {stats['uptime']:.2f} seconds\n")
            system_text.insert(tk.END, f"Memory Usage: {stats['memory_usage']:.2f} MB\n")
            system_text.insert(tk.END, f"Total Embeddings: {stats['total_embeddings']}\n")
            
            # Update Document Stats
            doc_text = self.stats_texts['documents']
            doc_text.delete(1.0, tk.END)
            doc_stats = stats['document_stats']
            doc_text.insert(tk.END, "=== Document Statistics ===\n\n")
            for key, value in doc_stats.items():
                if isinstance(value, dict):
                    doc_text.insert(tk.END, f"\n{key}:\n")
                    for k, v in value.items():
                        doc_text.insert(tk.END, f"  {k}: {v}\n")
                else:
                    doc_text.insert(tk.END, f"{key}: {value}\n")
            
            # Update Search Stats
            search_text = self.stats_texts['search']
            search_text.delete(1.0, tk.END)
            search_stats = stats['search_stats']
            search_text.insert(tk.END, "=== Search Statistics ===\n\n")
            for key, value in search_stats.items():
                search_text.insert(tk.END, f"{key}: {value}\n")
            
            # Update Chat Stats
            chat_text = self.stats_texts['chat']
            chat_text.delete(1.0, tk.END)
            chat_stats = stats['chat_stats']
            chat_text.insert(tk.END, "=== Chat Statistics ===\n\n")
            for key, value in chat_stats.items():
                chat_text.insert(tk.END, f"{key}: {value}\n")

            self.status_bar.config(text="Statistics refreshed successfully")

        except Exception as e:
            for text_widget in self.stats_texts.values():
                text_widget.delete(1.0, tk.END)
                text_widget.insert(tk.END, f"Error refreshing statistics: {str(e)}")
            self.status_bar.config(text=f"Error refreshing statistics: {str(e)}")

    def _toggle_auto_refresh(self):
        if self.auto_refresh_var.get():
            self._schedule_refresh()

    def _schedule_refresh(self):
        if self.auto_refresh_var.get():
            self._refresh_statistics()
            self.root.after(5000, self._schedule_refresh)

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleRAGInterface(root)
    root.mainloop()