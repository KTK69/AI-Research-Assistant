import os
import logging
import tempfile
import asyncio
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
from pathlib import Path

# HTTP and async imports
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

# LangChain and AI imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF processing
from pypdf import PdfReader

# Environment and utilities
from dotenv import load_dotenv
import google.generativeai as genai  # Updated Gemini SDK

# Load environment variables
load_dotenv()

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration class
class AppConfig:
    # Text processing
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
    MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", 5))
    
    # Performance
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 2))
    TIMEOUT = int(os.getenv("TIMEOUT", 60))
    
    # AI Configuration - Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = "gemini-1.5-flash"  # Updated model
    
    # File handling
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {".pdf"}

config = AppConfig()

# Global state management
class AppState:
    def __init__(self):
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vector_store: Optional[Chroma] = None
        self.processed_file: Optional[str] = None
        self.processing_status: str = "idle"
        self.file_hash: Optional[str] = None
        self.chunk_count: int = 0
        self.text_length: int = 0

app_state = AppState()

# Pydantic models
class QuestionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    
    question: str = Field(
        ..., 
        min_length=3, 
        max_length=500, 
        description="Research question"
    )
    max_chunks: Optional[int] = Field(
        default=config.MAX_CHUNKS, 
        ge=1, 
        le=10, 
        description="Maximum context chunks"
    )

class ProcessingResponse(BaseModel):
    status: str
    filename: str
    chunks: int
    text_length: int
    processing_time: float
    file_hash: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    context_chunks: int
    confidence_score: Optional[float] = None
    processing_time: float

class StatusResponse(BaseModel):
    processed_file: Optional[str]
    ai_model: str
    embeddings_model: str
    status: str
    processing_status: str
    performance_metrics: Dict[str, Any]
    error: Optional[str] = None

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""],
    length_function=len,
)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("🚀 Starting AI PDF Research Assistant with Gemini...")
    
    try:
        # Initialize embeddings model
        app_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✅ Embeddings model loaded successfully")
        
        # Initialize Gemini with new SDK
        if config.GEMINI_API_KEY:
            # Using new client-based initialization
            genai.configure(api_key=config.GEMINI_API_KEY)
            logger.info("✅ Gemini initialized successfully")
        else:
            logger.warning("⚠️ GEMINI_API_KEY not set. Gemini features will be disabled.")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        app_state.embeddings = None
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down application...")
    cleanup_directories()
    executor.shutdown(wait=True)
    logger.info("✅ Application shutdown complete")

def cleanup_directories():
    """Clean up temporary directories"""
    import shutil
    import glob
    
    for pattern in ["./chroma_db*", "./temp_*"]:
        for path in glob.glob(pattern):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.unlink(path)
                logger.info(f"🧹 Cleaned up {path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup {path}: {str(e)}")

# FastAPI app initialization
app = FastAPI(
    title="AI PDF Research Assistant with Gemini",
    version="4.0.0",
    description="Python 3.12 optimized AI-powered research assistant using Google Gemini",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions
def calculate_file_hash(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()[:16]

def validate_pdf_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")
    
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Invalid file type. Expected PDF")

async def extract_text_from_pdf_async(file_path: str) -> str:
    def _extract():
        try:
            reader = PdfReader(file_path)
            text_parts = []
            
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(f"\n--- Page {i+1} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to extract page {i+1}: {str(e)}")
            
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"❌ PDF extraction failed: {str(e)}")
            raise
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _extract)

async def query_gemini(prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    """Query Gemini model using new SDK"""
    if not config.GEMINI_API_KEY:
        return {
            "text": "Gemini API key not configured",
            "processing_time": 0,
            "success": False
        }
    
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            # New SDK usage pattern
            model = genai.GenerativeModel(config.GEMINI_MODEL)
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    top_p=0.9,
                    max_output_tokens=1024
                )
            )
            
            # New response structure
            if response.text:
                return {
                    "text": response.text,
                    "processing_time": time.time() - start_time,
                    "success": True
                }
            else:
                logger.warning("Gemini response contained no text")
                return {
                    "text": "No response generated from Gemini",
                    "processing_time": time.time() - start_time,
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"❌ Gemini query failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
    
    return {
        "text": "Failed to get response from Gemini after multiple attempts",
        "processing_time": time.time() - start_time,
        "success": False
    }

# API Routes
@app.get("/")
async def read_root():
    return {
        "message": "🚀 AI PDF Research Assistant with Gemini API v4.0",
        "status": "running",
        "python_version": "3.12",
        "features": ["gemini_ai", "rag_system"],
        "endpoints": {
            "upload": "/upload",
            "ask": "/ask", 
            "status": "/status",
            "clear": "/clear"
        }
    }

@app.post("/upload", response_model=ProcessingResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not app_state.embeddings:
        raise HTTPException(500, "Embeddings model not available")
    
    validate_pdf_file(file)
    start_time = time.time()
    app_state.processing_status = "processing"
    
    try:
        content = await file.read()
        
        if len(content) > config.MAX_FILE_SIZE:
            raise HTTPException(413, "File too large")
        
        file_hash = calculate_file_hash(content)
        
        # Check if file already processed
        if app_state.file_hash == file_hash and app_state.vector_store:
            logger.info(f"📋 File already processed: {file.filename}")
            return ProcessingResponse(
                status="already_processed",
                filename=file.filename,
                chunks=app_state.chunk_count,
                text_length=app_state.text_length,
                processing_time=time.time() - start_time,
                file_hash=file_hash
            )
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"🔄 Processing PDF: {file.filename}")
        
        try:
            # Extract text
            text_content = await extract_text_from_pdf_async(tmp_path)
            
            # Create text chunks
            chunks = text_splitter.split_text(text_content)
            
            if not chunks:
                raise ValueError("No text chunks created")
            
            # Create vector store
            persist_dir = f"./chroma_db_{file_hash}"
            app_state.vector_store = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: Chroma.from_texts(
                    texts=chunks,
                    embedding=app_state.embeddings,
                    persist_directory=persist_dir
                )
            )
            
            # Update state
            app_state.processed_file = file.filename
            app_state.processing_status = "completed"
            app_state.file_hash = file_hash
            app_state.chunk_count = len(chunks)
            app_state.text_length = len(text_content)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Processed {file.filename} in {processing_time:.2f}s")
            
            return ProcessingResponse(
                status="success",
                filename=file.filename,
                chunks=len(chunks),
                text_length=len(text_content),
                processing_time=processing_time,
                file_hash=file_hash
            )
            
        finally:
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup temp file: {e}")
    
    except Exception as e:
        app_state.processing_status = "error"
        logger.error(f"❌ Processing failed: {str(e)}")
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not app_state.vector_store:
        raise HTTPException(400, "No document uploaded")
    
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Question: {request.question[:50]}...")
        
        # Retrieve relevant chunks
        docs_with_scores = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: app_state.vector_store.similarity_search_with_score(
                request.question, 
                k=request.max_chunks
            )
        )
        
        if not docs_with_scores:
            return AnswerResponse(
                question=request.question,
                answer="No relevant information found",
                sources=[],
                context_chunks=0,
                processing_time=time.time() - start_time
            )
        
        # Build context
        context_parts = []
        sources = []
        total_score = sum(score for _, score in docs_with_scores)
        avg_score = total_score / len(docs_with_scores)
        
        for i, (doc, score) in enumerate(docs_with_scores):
            relevance = max(0, 1 - score)
            context_parts.append(
                f"Context {i+1} (relevance: {relevance:.2f}):\n{doc.page_content}"
            )
            sources.append(f"Document section {i+1}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt
        prompt = f"""You are an expert AI research assistant. Provide precise answers based ONLY on the provided context.

Document Context:
{context}

Question: {request.question}

Guidelines:
- Base answer ONLY on provided context
- Be comprehensive yet concise
- Use specific evidence from document
- Structure response clearly
- If information is insufficient, state this
- Maintain academic rigor

Response:"""
        
        # Get AI response
        ai_result = await query_gemini(prompt)
        
        processing_time = time.time() - start_time
        confidence_score = max(0, 1 - avg_score) if avg_score < 1 else None
        
        logger.info(f"✅ Answered in {processing_time:.2f}s")
        
        return AnswerResponse(
            question=request.question,
            answer=ai_result["text"],
            sources=sources,
            context_chunks=len(docs_with_scores),
            confidence_score=confidence_score,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"❌ Query failed: {str(e)}")
        raise HTTPException(500, f"Query error: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        processed_file=app_state.processed_file,
        ai_model=f"Gemini {config.GEMINI_MODEL}" if config.GEMINI_API_KEY else "Not configured",
        embeddings_model="all-MiniLM-L6-v2" if app_state.embeddings else "Not available",
        status="ready" if app_state.vector_store else "awaiting_upload",
        processing_status=app_state.processing_status,
        performance_metrics={
            "chunks_processed": app_state.chunk_count,
            "document_size": app_state.text_length,
            "python_version": "3.12",
            "chunk_size": config.CHUNK_SIZE,
        },
        error=None if app_state.embeddings and config.GEMINI_API_KEY else "Missing configuration"
    )

@app.delete("/clear")
async def clear_data():
    logger.info("🧹 Clearing application data...")
    
    # Reset state
    app_state.vector_store = None
    app_state.processed_file = None
    app_state.processing_status = "idle"
    app_state.file_hash = None
    app_state.chunk_count = 0
    app_state.text_length = 0
    
    # Cleanup directories
    cleanup_directories()
    
    logger.info("✅ Data cleared successfully")
    return {
        "status": "cleared", 
        "message": "All data cleared",
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "embeddings_loaded": app_state.embeddings is not None,
        "gemini_configured": config.GEMINI_API_KEY is not None,
        "document_loaded": app_state.vector_store is not None
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )