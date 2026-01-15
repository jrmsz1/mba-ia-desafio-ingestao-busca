import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # "openai" ou "google"

def get_embeddings():
    """Retorna o modelo de embeddings baseado no provider configurado"""
    provider = EMBEDDING_PROVIDER.lower()
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        print(f"Using OpenAI embeddings: {model}")
        return OpenAIEmbeddings(model=model)
    
    elif provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        print(f"Using Google embeddings: {model}")
        return GoogleGenerativeAIEmbeddings(model=model)
    
    else:
        raise RuntimeError(f"Unknown EMBEDDING_PROVIDER: {provider}. Use 'openai' or 'google'")


def ingest_pdf():
    # Validação das variáveis de ambiente
    for k in ("DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
        if not os.getenv(k):
            raise RuntimeError(f"Environment variable {k} is not set")
    
    if not PDF_PATH:
        raise RuntimeError("Environment variable PDF_PATH is not set")
    
    # Carregamento do PDF
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"Loading PDF: {pdf_path}")
    docs = PyPDFLoader(str(pdf_path)).load()
    
    # Divisão em chunks
    print("Splitting documents into chunks...")
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150, 
        add_start_index=False
    ).split_documents(docs)
    
    if not splits:
        print("No documents to process")
        raise SystemExit(0)
    
    print(f"Created {len(splits)} chunks")
    
    # Limpeza dos metadados
    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in splits
    ]
    
    # Geração dos IDs
    ids = [f"doc-{i}" for i in range(len(enriched))]
    
    # Criação dos embeddings
    print("Creating embeddings...")
    embeddings = get_embeddings()
    
    # Conexão com o PostgreSQL + pgVector
    print("Connecting to PostgreSQL with pgVector...")
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )
    
    # Armazenamento dos documentos
    print("Storing documents in database...")
    store.add_documents(documents=enriched, ids=ids)
    
    print(f"Successfully ingested {len(enriched)} chunks into the database")


if __name__ == "__main__":
    ingest_pdf()