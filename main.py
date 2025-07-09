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

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import easyocr
import cv2
import numpy as np
import tempfile
import os
import re
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NID Parser API",
    description="API for extracting structured information from NID images using OCR",
    version="1.0.0"
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
    """Extract name from text. Name will come after 'Name' and may include abbreviations like 'MD.' before all-caps words."""
    # Look for 'Name' followed by optional abbreviation (MD., MD, MD-, etc.) and all-caps words
    name_pattern = r'Name[:\s]+((?:MD[\.,\-]?\s*)?(?:[A-Z][A-Z]+(?:\s+|\.|$))+)'  # MD. + all-caps words
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
    # Fallback: try to find the longest all-caps sequence, including 'MD.' if present before
    fallback_pattern = r'(MD[\.,-]?\s*)?([A-Z]{2,}(?:\s+[A-Z]{2,})*)'
    all_caps = re.findall(fallback_pattern, text)
    if all_caps:
        # Find the longest match
        best = max(all_caps, key=lambda x: len((x[0] + x[1]).strip()))
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

        # Load and downscale image if large
        img = cv2.imread(image_path)
        max_dim = 800
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            # Save to temp file for OCR
            temp_downscaled = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_downscaled.name, img)
            ocr_image_path = temp_downscaled.name
        else:
            ocr_image_path = image_path

        # Perform OCR (detail=0 for faster text extraction)
        logger.info(f"Performing OCR on {ocr_image_path}")
        results = reader.readtext(ocr_image_path, detail=0)

        # Clean up temp downscaled file if used
        if ocr_image_path != image_path and os.path.exists(ocr_image_path):
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

@app.post("/extract-nid-info/")
async def extract_nid_info(file: UploadFile = File(...)):
    """
    Extract structured information from NID image.
    
    Upload an image file and get back structured information including:
    - name: detected name
    - dob: detected date of birth  
    - nid: detected NID number
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        try:
            # Write uploaded file to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Perform OCR analysis
            result = perform_ocr_analysis(temp_file.name)
            
            return JSONResponse(content=result)
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "NID Parser API",
        "version": "1.0.0",
        "endpoints": {
            "extract_nid_info": "/extract-nid-info/",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 