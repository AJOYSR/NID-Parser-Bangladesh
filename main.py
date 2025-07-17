#--------------------------------------------------------------------
#Created By "AJOY SARKER"
#Email: ajoysr.official@gmail.com
#Github: @ajoysr
#LinkedIn: @ajoysrju
#--------------------------------------------------------------------

import ssl
import urllib.request

# Fix SSL certificate issues on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:   
    ssl._create_default_https_context = _create_unverified_https_context

# Fix PIL.Image.ANTIALIAS compatibility issue
try:
    from PIL import Image
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
except ImportError:
    pass

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import easyocr
import cv2
import numpy as np
import tempfile
import os
import re
from datetime import datetime
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add PDF/image support
try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None


# Pydantic models for API documentation
class NIDInfoResponse(BaseModel):
    """Response model for NID information extraction."""
    name: str = "Not detected"
    dob: str = "Not detected"
    nid: str = "Not detected"
    extracted_text: str = ""

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str

class RootResponse(BaseModel):
    """Response model for root endpoint."""
    message: str
    version: str
    endpoints: Dict[str, str]

app = FastAPI(
    title="NID Parser API",
    description="""
    ## NID Parser API
    
    A powerful API for extracting structured information from National ID (NID) images using Optical Character Recognition (OCR).
    """,
    version="1.0.0",
    contact={
        "name": "AJOY SARKER",
        "email": "ajoysr.official@gmail.com",
        "url": "https://github.com/ajoysr"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.on_event("startup")
async def startup_event():
    """Pre-initialize the EasyOCR reader on startup."""
    logger.info("Starting up NID Parser API...")
    initialize_reader()
    logger.info("Startup complete - EasyOCR reader ready")

# Global reader instance for better performance
reader = None

def initialize_reader():
    """Initialize the EasyOCR reader with English language support and a smaller recognition network for speed."""
    global reader
    if reader is None:
        logger.info("Initializing EasyOCR reader...")
        # Use a smaller recognition network for faster inference
        reader = easyocr.Reader(['en'], gpu=False, download_enabled=True, model_storage_directory='./models', recog_network='english_g2')
        logger.info("EasyOCR reader initialized successfully")
    return reader

def extract_date_of_birth(text: str) -> Optional[str]:
    """Extract date of birth from text using various patterns."""
    # Common date patterns
    date_patterns = [
        r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b',  # DD/MM/YYYY or DD-MM-YYYY
        r'\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b',  # YYYY/MM/DD
        r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{2,4})\b',  # DD Month YYYY
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{2,4})\b',  # Month DD, YYYY
    ]
    
    for pattern in date_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            return match.group(0)  # Return the full matched string
    return None


def extract_nid_number(text: str) -> Optional[str]:
    """Extract NID number from text."""
    # NID number patterns (adjust based on your country's format)
    nid_patterns = [
        r'\b(\d{2,6}(?:\s\d{2,6}){2,5})\b',  # e.g., 600 458 9963 or similar
        r'\b(\d{10,17})\b',  # 10-17 digit numbers
        r'\bNID[:\s]*(\d[\d\s]+)\b',  # NID: followed by numbers (with spaces)
        r'\bID[:\s]*(\d[\d\s]+)\b',   # ID: followed by numbers (with spaces)
        r'\bNational\s+ID[:\s]*(\d[\d\s]+)\b',  # National ID: followed by numbers (with spaces)
    ]
    
    for pattern in nid_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Return the first match, stripped of leading/trailing whitespace
            return matches[0].strip()
    return None

def extract_name(text: str) -> Optional[str]:
    """Extract name from text. Prioritize patterns that look like actual names."""
    # First, look for "MD:" followed by name (most specific pattern)
    # Look for "MD:" followed by name, but stop before "Data of Birth" or similar
    md_pattern = r'MD:\s*([A-Z][A-Z]+(?:\s+[A-Z]+)*?)(?=\s+(?:Data|Date|of|Birth|0t|Birth|ara|@ue|\d|$))'
    match = re.search(md_pattern, text)
    if match:
        name = match.group(1).strip()
        # Clean up the name - remove any trailing single letters that might be artifacts
        name_parts = name.split()
        if len(name_parts) > 1 and len(name_parts[-1]) == 1:
            # Remove single letter at the end (like "D" in "ZAKIR HOSSAIN D")
            name_parts = name_parts[:-1]
        return f"MD: {' '.join(name_parts)}"
    
    # Look for "Name:" followed by name
    name_pattern = r'Name[:\s]+((?:MD[\.,\-]?\s*)?(?:[A-Z][A-Z]+(?:\s+|\.|$))+)'
    match = re.search(name_pattern, text)
    if match:
        name = match.group(1)
        # Split into words, keep 'MD.' or similar and all-caps words, join back
        words = re.split(r'\s+', name)
        filtered = []
        for w in words:
            if re.fullmatch(r'MD[\.,-]?', w):
                filtered.append('MD.')
            elif re.fullmatch(r'[A-Z]+', w):
                filtered.append(w)
        if filtered:
            return ' '.join(filtered)
    
    # Look for "Data of Birth" or "Date of Birth" context - name often appears before this
    dob_context_pattern = r'([A-Z][A-Z]+(?:\s+[A-Z]+)*)\s+(?:Data|Date)\s+of\s+Birth'
    match = re.search(dob_context_pattern, text)
    if match:
        name = match.group(1).strip()
        # Check if it looks like a reasonable name (not too long, not common words)
        if len(name.split()) <= 3 and not any(word in ['SCREENSHOT', 'RECORDER', 'CHROME', 'EXTENSION'] for word in name.split()):
            return name
    
    # Fallback: try to find all-caps sequences, but filter out common non-name words
    fallback_pattern = r'(MD[\.,-]?\s*)?([A-Z]{2,}(?:\s+[A-Z]{2,})*)'
    all_caps = re.findall(fallback_pattern, text)
    if all_caps:
        # Filter out common non-name words and find the best match
        filtered_matches = []
        for prefix, name_part in all_caps:
            name_words = name_part.strip().split()
            # Filter out common non-name words
            filtered_words = [word for word in name_words 
                            if word not in ['SCREENSHOT', 'RECORDER', 'CHROME', 'EXTENSION', 'DEVELOPMENT', 
                                          'INTERVIEW', 'COMPANY', 'REPOSITORIES', 'RESEARCH', 'TRANSLATE',
                                          'FEEDBACK', 'OPTIONS', 'PEOPLE', 'REPUBLIC']]
            if filtered_words:
                filtered_matches.append((prefix, ' '.join(filtered_words)))
        
        if filtered_matches:
            # Find the longest reasonable match
            best = max(filtered_matches, key=lambda x: len((x[0] + x[1]).strip()))
            name_parts = []
            if best[0]:
                name_parts.append('MD.')
            if best[1]:
                name_parts.append(best[1].strip())
            if name_parts:
                return ' '.join(name_parts)
    
    return None

def perform_ocr_analysis(image_path: str) -> Dict[str, str]:
    """Perform OCR analysis and extract structured information."""
    try:
        # Initialize reader
        reader = initialize_reader()

        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Unable to read image file")
        
        # Convert to grayscale for better OCR accuracy
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        
        
        # Downscale if large
        max_dim = 800
        h, w = gray_img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            processed_img = cv2.resize(gray_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            processed_img = gray_img

        # Save processed image to temp file for OCR
        temp_processed = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_processed.name, processed_img)
        ocr_image_path = temp_processed.name

        # Perform OCR (detail=0 for faster text extraction)
        logger.info(f"Performing OCR on preprocessed image: {ocr_image_path}")
        results = reader.readtext(ocr_image_path, detail=0)

        # Clean up temp file
        if os.path.exists(ocr_image_path):
            os.unlink(ocr_image_path)

        # Extract all text
        all_text = ' '.join(results)
        logger.info(f"Extracted text: {all_text[:200]}...")  # Log first 200 chars

        # Extract structured information
        name = extract_name(all_text)
        dob = extract_date_of_birth(all_text)
        nid = extract_nid_number(all_text)

        return {
            "name": name or "Not detected",
            "dob": dob or "Not detected", 
            "nid": nid or "Not detected",
            "extracted_text": all_text
        }

    except Exception as e:
        logger.error(f"Error during OCR analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

def extract_fields_by_type(doc_type: str, text: str) -> dict:
    """Extract fields based on document type."""
    # Default all fields to None
    details = {
        "name": None,
        "date_of_birth": None,
        "nid_number": None,
        "bo_id": None,
        "tin_number": None
    }
    if doc_type == 'NID':
        details["name"] = extract_name(text)
        details["date_of_birth"] = extract_date_of_birth(text)
        details["nid_number"] = extract_nid_number(text)
    elif doc_type == 'BO':
        # Example: BO ID pattern (customize as needed)
        bo_pattern = r'BO\s*Account\s*Number\s*[:\-]?\s*(120\d{3}\s\d{10})'
        match = re.search(bo_pattern, text, re.IGNORECASE)
        details["bo_id"] = match.group(1) if match else None
    elif doc_type == 'TIN':
        # Example: TIN pattern (customize as needed)
        tin_pattern = r'\bTIN[\s:-]*(\d{9,12})\b'
        match = re.search(tin_pattern, text, re.IGNORECASE)
        details["tin_number"] = match.group(1) if match else None
    # Add more types as needed
    return details

async def extract_text_from_file(file: UploadFile, content: bytes) -> str:
    """Extract text from image or PDF file."""
    if file.content_type == 'application/pdf':
        if convert_from_bytes is None:
            raise HTTPException(status_code=500, detail="pdf2image is not installed")
        # Convert PDF to images
        images = convert_from_bytes(content)
        all_text = []
        for img in images:
            # Save PIL image to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
                img.save(temp_img.name, 'JPEG')
                text = perform_ocr_analysis(temp_img.name)["extracted_text"]
                all_text.append(text)
                os.unlink(temp_img.name)
        return ' '.join(all_text)
    elif file.content_type.startswith('image/'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
            temp_img.write(content)
            temp_img.flush()
            text = perform_ocr_analysis(temp_img.name)["extracted_text"]
            os.unlink(temp_img.name)
            return text
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only images and PDFs are allowed.")

@app.post("/extract-nid-info/", summary="Extract Document Information", tags=["Document Processing"])
async def extract_nid_info(
    type: str = Form(..., description="Type of document: NID, BO, TIN"),
    file: UploadFile = File(..., description="Image or PDF file of the document")
):
    try:
        logger.info(f"Received request - filename: {file.filename}, content_type: {file.content_type}, type: {type}")
        # Validate file type
        if not file.content_type or (not file.content_type.startswith('image/') and file.content_type != 'application/pdf'):
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image or PDF")
        # Validate file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(status_code=400, detail="File size too large. Maximum size is 10MB")
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        # Extract text from file (image or PDF)
        all_text = await extract_text_from_file(file, content)
        print(all_text)
        # Extract fields based on type
        details = extract_fields_by_type(type, all_text)
        
        # Always return all fields, set missing to None
        response = {
            "type": type,
            "details": details,
            "raw_text": all_text
        }
        return JSONResponse(content=response)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in extract_nid_info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/", 
         response_model=RootResponse,
         summary="API Information",
         description="""
         Get basic information about the NID Parser API.
         
         This endpoint provides:
         - API name and version
         - Available endpoints
         - Basic usage information
         """,
         responses={
             200: {
                 "description": "API information retrieved successfully",
                 "content": {
                     "application/json": {
                         "example": {
                             "message": "NID Parser API",
                             "version": "1.0.0",
                             "endpoints": {
                                 "extract_nid_info": "/extract-nid-info/",
                                 "docs": "/docs",
                                 "health": "/health"
                             }
                         }
                     }
                 }
             }
         },
         tags=["API Information"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "NID Parser API",
        "version": "1.0.0",
        "endpoints": {
            "extract_nid_info": "/extract-nid-info/",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health", 
         response_model=HealthResponse,
         summary="Health Check",
         description="""
         Check the health status of the NID Parser API.
         
         This endpoint is useful for:
         - Monitoring system health
         - Load balancer health checks
         - Verifying API availability
         
         Returns the current status and timestamp.
         """,
         responses={
             200: {
                 "description": "API is healthy",
                 "content": {
                     "application/json": {
                         "example": {
                             "status": "healthy",
                             "timestamp": "2024-01-15T10:30:00.123456"
                         }
                     }
                 }
             }
         },
         tags=["Monitoring"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
