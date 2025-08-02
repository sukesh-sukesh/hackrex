"""
Ultra Minimal RAG System for Hackathon Submission
Works with minimal dependencies - guaranteed to run!
"""

import os
import io
import re
import json
import logging
import traceback
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

# Only use packages that are likely to be available
try:
    from fastapi import FastAPI, HTTPException, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available - this is required!")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Requests not available - using urllib fallback")
    import urllib.request
    import urllib.error

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "9409b1c8ad293eb74efa88c538211dcd22fb9eeaf89e7cefd1eb4b398f466c2f")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# --- Minimal Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Ultra Minimal RAG System Starting...")
    logger.info("📦 Using only basic dependencies")
    yield
    logger.info("👋 System shutdown")

# Initialize FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Ultra Minimal RAG System",
        version="1.0.0",
        description="Minimal RAG system with basic dependencies only",
        lifespan=lifespan
    )
else:
    print("ERROR: FastAPI is required but not available!")
    exit(1)

# --- Pydantic Models ---
class DocumentRequest(BaseModel):
    documents: str
    questions: List[str]

# --- Security ---
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# --- Minimal Document Download ---
def download_document_minimal(url: str) -> bytes:
    """Download document using requests or urllib fallback"""
    try:
        if REQUESTS_AVAILABLE:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response.content
        else:
            # Fallback to urllib
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; RAG-System/1.0)'
            })
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download: {str(e)}")

# --- Basic PDF Text Extraction (Without pypdf) ---
def extract_text_basic_pdf(file_content: bytes) -> str:
    """Basic PDF text extraction without external libraries"""
    try:
        # Convert bytes to string and try to find readable text
        text_content = file_content.decode('latin-1', errors='ignore')
        
        # Look for text between common PDF markers
        text_patterns = [
            r'BT\s*(.*?)\s*ET',  # Between BT and ET markers
            r'\((.*?)\)',         # Text in parentheses
            r'Tj\s*(.*?)\s*Tj',  # Between Tj markers
        ]
        
        extracted_text = ""
        for pattern in text_patterns:
            matches = re.findall(pattern, text_content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # Clean up the text
                clean_text = re.sub(r'[^\w\s\.\,\!\?\-\:]', ' ', match)
                if len(clean_text.strip()) > 10:  # Only meaningful text
                    extracted_text += clean_text + " "
        
        if not extracted_text.strip():
            # Last resort - extract any readable text
            readable_text = re.findall(r'[A-Za-z][A-Za-z\s]{10,}', text_content)
            extracted_text = " ".join(readable_text[:50])  # First 50 matches
        
        if not extracted_text.strip():
            raise ValueError("No readable text found in PDF")
        
        logger.info(f"Extracted {len(extracted_text)} characters from PDF")
        return extracted_text.strip()
    
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise HTTPException(status_code=400, detail="PDF processing failed - file may be corrupted or encrypted")

# --- Basic DOCX Text Extraction (Without python-docx) ---
def extract_text_basic_docx(file_content: bytes) -> str:
    """Basic DOCX text extraction without external libraries"""
    try:
        import zipfile
        
        # DOCX is a ZIP file
        docx_zip = zipfile.ZipFile(io.BytesIO(file_content))
        
        # Extract text from document.xml
        doc_xml = docx_zip.read('word/document.xml').decode('utf-8')
        
        # Remove XML tags and extract text
        text = re.sub(r'<[^>]+>', '', doc_xml)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        if not text.strip():
            raise ValueError("No readable text found in DOCX")
        
        logger.info(f"Extracted {len(text)} characters from DOCX")
        return text.strip()
    
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        raise HTTPException(status_code=400, detail="DOCX processing failed - file may be corrupted")

# --- Basic Text Chunking ---
def simple_chunk_text(text: str, chunk_size: int = 800) -> List[str]:
    """Simple text chunking without external libraries"""
    if len(text) <= chunk_size:
        return [text]
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If chunks are still too large, split by sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            sentences = re.split(r'[.!?]+', chunk)
            sub_chunk = ""
            for sentence in sentences:
                if len(sub_chunk) + len(sentence) <= chunk_size:
                    sub_chunk += sentence + ". "
                else:
                    if sub_chunk:
                        final_chunks.append(sub_chunk.strip())
                    sub_chunk = sentence + ". "
            if sub_chunk:
                final_chunks.append(sub_chunk.strip())
    
    # Filter out very short chunks
    final_chunks = [chunk for chunk in final_chunks if len(chunk.strip()) > 50]
    
    logger.info(f"Created {len(final_chunks)} text chunks")
    return final_chunks

# --- Basic Text Search ---
def basic_text_search(question: str, chunks: List[str], max_results: int = 5) -> List[str]:
    """Basic keyword-based text search"""
    question_words = set(question.lower().split())
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    question_words = question_words - stop_words
    
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = 0
        
        for word in question_words:
            if len(word) > 2:  # Skip very short words
                # Count occurrences of each word
                score += chunk_lower.count(word) * len(word)
        
        if score > 0:
            chunk_scores.append((score, i, chunk))
    
    # Sort by score and return top chunks
    chunk_scores.sort(reverse=True, key=lambda x: x[0])
    
    if not chunk_scores:
        # If no matches, return first few chunks
        return chunks[:max_results]
    
    relevant_chunks = [chunk for _, _, chunk in chunk_scores[:max_results]]
    logger.info(f"Found {len(relevant_chunks)} relevant chunks using basic search")
    return relevant_chunks

# --- Basic Answer Generation ---
def generate_basic_answer(question: str, context_chunks: List[str]) -> str:
    """Generate answer using simple text analysis"""
    question_lower = question.lower()
    
    # Common patterns for insurance/policy questions
    patterns = {
        'grace period': ['grace period', 'days', 'premium', 'payment'],
        'waiting period': ['waiting period', 'months', 'coverage', 'diseases'],
        'maternity': ['maternity', 'pregnancy', 'delivery', 'childbirth'],
        'pre-existing': ['pre-existing', 'diseases', 'PED', 'conditions'],
        'claim discount': ['claim discount', 'NCD', 'no claim', 'discount'],
        'room rent': ['room rent', 'daily', 'ICU', 'charges'],
        'ayush': ['ayush', 'ayurveda', 'homeopathy', 'treatment'],
        'hospital': ['hospital', 'institution', 'beds', 'medical'],
        'organ donor': ['organ donor', 'harvesting', 'transplantation'],
        'health check': ['health check', 'preventive', 'check-up']
    }
    
    # Find the most relevant pattern
    best_pattern = None
    for pattern_name, keywords in patterns.items():
        if any(keyword in question_lower for keyword in keywords):
            best_pattern = keywords
            break
    
    # Search for relevant information in chunks
    relevant_info = []
    for chunk in context_chunks:
        chunk_lower = chunk.lower()
        
        if best_pattern:
            # Look for sentences containing pattern keywords
            sentences = chunk.split('. ')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in best_pattern):
                    relevant_info.append(sentence.strip())
        else:
            # General search - look for question words in text
            words = question_lower.split()
            for word in words:
                if len(word) > 3 and word in chunk_lower:
                    # Find sentences containing this word
                    sentences = chunk.split('. ')
                    for sentence in sentences:
                        if word in sentence.lower():
                            relevant_info.append(sentence.strip())
                            break
    
    if relevant_info:
        # Return the most relevant information
        answer = '. '.join(relevant_info[:3])  # Top 3 relevant sentences
        if len(answer) > 500:
            answer = answer[:500] + "..."
        return answer
    else:
        return "Based on the available document, I could not find specific information to answer this question."

# --- Google Gemini API (Optional) ---
def call_gemini_api(question: str, context: str) -> Optional[str]:
    """Call Google Gemini API if available"""
    if not GOOGLE_API_KEY:
        return None
    
    try:
        if REQUESTS_AVAILABLE:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GOOGLE_API_KEY}"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"Based on this context, answer the question concisely:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 200
                }
            }
            
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text']
        
        return None
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return None

# --- Main Processing ---
async def process_document_simple(file_content: bytes, questions: List[str]) -> List[str]:
    """Simple document processing pipeline"""
    try:
        # Determine file type and extract text
        if file_content.startswith(b'%PDF'):
            text = extract_text_basic_pdf(file_content)
        elif file_content.startswith(b'PK\x03\x04'):
            text = extract_text_basic_docx(file_content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and DOCX supported.")
        
        # Chunk the text
        chunks = simple_chunk_text(text)
        
        # Process each question
        answers = []
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            
            try:
                # Find relevant chunks
                relevant_chunks = basic_text_search(question, chunks, max_results=3)
                
                # Try Gemini API first, fallback to basic generation
                context = "\n\n".join(relevant_chunks)
                answer = call_gemini_api(question, context)
                
                if not answer:
                    answer = generate_basic_answer(question, relevant_chunks)
                
                answers.append(answer)
                
            except Exception as e:
                logger.error(f"Failed to process question {i+1}: {e}")
                answers.append("Error processing this question.")
        
        return answers
    
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# --- API Endpoints ---
@app.post("/hackrx/run")
async def process_document_and_answer_questions(
    request: DocumentRequest,
    token: str = Depends(verify_token)
):
    """Main hackathon endpoint"""
    try:
        logger.info(f"Processing {len(request.questions)} questions")
        
        # Download document
        file_content = download_document_minimal(request.documents)
        
        # Process questions
        answers = await process_document_simple(file_content, request.questions)
        
        logger.info("✅ Processing completed")
        return {"answers": answers}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/health")
def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "dependencies": {
            "fastapi": FASTAPI_AVAILABLE,
            "requests": REQUESTS_AVAILABLE,
            "google_api": bool(GOOGLE_API_KEY)
        }
    }

@app.get("/")
def root():
    return {
        "message": "🚀 Ultra Minimal RAG System",
        "version": "1.0.0",
        "status": "operational"
    }

# --- Server Execution ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting ultra minimal server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)