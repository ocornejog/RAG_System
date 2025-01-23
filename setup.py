import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        console.print("[bold red]Error: Python 3.7 or higher is required[/bold red]")
        sys.exit(1)

def install_requirements():
    """Install required packages."""
    requirements = [
        "fastapi",
        "uvicorn",
        "sentence-transformers",
        "numpy",
        "rich",
        "requests",
        "python-dotenv",
        "google-generativeai>=0.3.0",
        "watchdog",  # For file monitoring
        "python-multipart",
        "typing-extensions",
        "pydantic",
        "psutil",
        "torch>=1.6.0"
    ]
    
    console.print("[bold blue]Installing required packages...[/bold blue]")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            console.print(f"[green]✓[/green] {package} installed successfully")
        except subprocess.CalledProcessError:
            console.print(f"[red]✗[/red] Failed to install {package}")
            return False
    return True

def create_env_file():
    """Create .env file if it doesn't exist."""
    env_path = Path(".env")
    if not env_path.exists():
        console.print("\n[bold yellow]Creating .env file...[/bold yellow]")
        google_key = console.input("Enter your Google API key (press Enter to skip): ")
        
        with env_path.open("w") as f:
            if google_key:
                f.write(f"GOOGLE_API_KEY={google_key}\n")
            else:
                f.write("GOOGLE_API_KEY=your_key_here\n")
        
        console.print("[green]✓[/green] Created .env file")

def check_dependencies():
    """Check if all required system dependencies are available."""
    dependencies = {
        "python": "python --version",
        "pip": "pip --version"
    }
    
    for dep, command in dependencies.items():
        try:
            subprocess.check_call(command.split(), stdout=subprocess.DEVNULL)
            console.print(f"[green]✓[/green] {dep} is available")
        except:
            console.print(f"[red]✗[/red] {dep} is not available")
            return False
    return True

def main():
    """Main setup function."""
    console.print(Panel.fit(
        "[bold blue]RAG System Setup[/bold blue]\n\n"
        "This script will set up the RAG system environment."
    ))
    
    # Check system dependencies
    if not check_dependencies():
        console.print("[bold red]Setup failed: Missing system dependencies[/bold red]")
        return
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    if not install_requirements():
        console.print("[bold red]Setup failed: Could not install requirements[/bold red]")
        return
    
    # Create .env file
    create_env_file()
    
    # Print success message and instructions
    console.print(Panel(
        "[bold green]Setup completed successfully![/bold green]\n\n"
        "To start using the system:\n\n"
        "1. Make sure your Google API key is set in the .env file\n"
        "2. Start the RAG server:\n"
        "   [bold]python rag_system.py[/bold]\n\n"
        "3. In another terminal, start the interface:\n"
        "   [bold]python demo.py[/bold]\n\n"
        "4. The system will automatically create necessary directories\n"
        "   when started and monitor for document changes.\n\n"
        "Note: Add your documents to the 'documents' directory once the system is running.",
        title="Next Steps",
        border_style="green"
    ))

if __name__ == "__main__":
    main()