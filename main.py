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
from enum import Enum
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


# Enum for document types
class DocumentType(str, Enum):
    """Enum for supported document types."""
    NID = "NID"
    BO = "BO"
    TIN = "TIN"


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

# Month names as they are printed on English-language cards.
MONTH_NAMES = r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec'

# Date candidates, paired with the order their groups appear in.
#
# The year group deliberately has no trailing word boundary. OCR regularly
# appends a stray digit to the year ("23 Jan 19941" on a perfectly legible
# card), and requiring a boundary there threw the whole date away. Matches that
# are not real dates are filtered out by _is_plausible_date() instead.
DATE_PATTERNS = (
    # 15/03/1990, 15-03-1990, 15.03.1990 (day first)
    (re.compile(r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})'), 'dmy'),
    # 1990/03/15 (year first)
    (re.compile(r'\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})'), 'ymd'),
    # 15 Mar 1990
    (re.compile(r'\b(\d{1,2})\s+(' + MONTH_NAMES + r')[a-z]*\s+(\d{2,4})', re.IGNORECASE), 'dmy_name'),
    # Mar 15, 1990
    (re.compile(r'\b(' + MONTH_NAMES + r')[a-z]*\s+(\d{1,2}),?\s+(\d{2,4})', re.IGNORECASE), 'mdy_name'),
)

# The "Date of Birth" label, and the OCR variants seen on real cards:
# "Data of Birth", "Date ol Birth", "Date 0f Birth", "DOB", "D.O.B."
DOB_LABEL_PATTERN = re.compile(r'\b(?:dat[ae]\s*[o0][fl]?\s*b[il1]rth|d\.?o\.?b)\b', re.IGNORECASE)

# How far after the label to look for the date itself.
DOB_LABEL_WINDOW = 60


def _is_plausible_year(raw: str) -> bool:
    """Check that a matched year could belong to a date of birth.

    Three digits means OCR truncated the year ("23 Jan 199"). Returning that as
    a date of birth is worse than returning nothing, so it is rejected.
    """
    if len(raw) == 2:
        return True  # Ambiguous but common on older documents.
    if len(raw) == 4:
        return 1900 <= int(raw) <= datetime.now().year
    return False


def _is_plausible_date(kind: str, groups) -> bool:
    """Reject matches whose day, month or year cannot be real."""
    if kind == 'dmy':
        day, month, year = int(groups[0]), int(groups[1]), groups[2]
    elif kind == 'ymd':
        year, month, day = groups[0], int(groups[1]), int(groups[2])
    elif kind == 'dmy_name':
        day, month, year = int(groups[0]), 1, groups[2]  # Month matched by name.
    else:  # mdy_name
        day, month, year = int(groups[1]), 1, groups[2]  # Month matched by name.
    return 1 <= day <= 31 and 1 <= month <= 12 and _is_plausible_year(year)


def _find_date_candidates(text: str):
    """Return (position, matched text) for every plausible date, in reading order."""
    candidates = []
    for pattern, kind in DATE_PATTERNS:
        for match in pattern.finditer(text):
            if _is_plausible_date(kind, match.groups()):
                candidates.append((match.start(), match.group(0)))
    candidates.sort(key=lambda item: item[0])
    return candidates


def extract_date_of_birth(text: str) -> Optional[str]:
    """Extract date of birth from text, preferring a date that follows its label.

    Documents carry other dates - an issue date, an expiry date, a registration
    date - so the first date in the text is not necessarily the right one. When
    OCR read the "Date of Birth" label clearly enough to find, the date that
    follows it wins. Returns None when nothing plausible is found.
    """
    candidates = _find_date_candidates(text)
    if not candidates:
        return None

    for label in DOB_LABEL_PATTERN.finditer(text):
        for start, value in candidates:
            if label.end() <= start <= label.end() + DOB_LABEL_WINDOW:
                return value

    return candidates[0][1]


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

# All-caps words that turn up beside a name but are not part of it: card
# labels, and browser chrome when the document was captured as a screenshot.
NON_NAME_WORDS = frozenset({
    'SCREENSHOT', 'RECORDER', 'CHROME', 'EXTENSION', 'DEVELOPMENT',
    'INTERVIEW', 'COMPANY', 'REPOSITORIES', 'RESEARCH', 'TRANSLATE',
    'FEEDBACK', 'OPTIONS', 'PEOPLE', 'REPUBLIC',
    'NATIONAL', 'CARD', 'GOVERNMENT', 'BANGLADESH', 'NAME', 'DATE', 'DATA',
    'BIRTH', 'BLOOD', 'GROUP', 'PLACE', 'ISSUE', 'NID', 'ID', 'NO',
})

# "MD: ZAKIR HOSSAIN" or "MD. ZAKIR HOSSAIN". The card prints the honorific
# with a period, but OCR frequently reads it as a colon, so both are accepted
# and whichever was read is preserved in the result.
#
# The name runs on for as long as the words stay all-caps, which stops it at
# the next field label or at OCR noise. Anchoring on a following label instead
# used to drop the honorific whenever that label was misread.
MD_NAME_PATTERN = re.compile(r'\bMD\s*([:.])\s*([A-Z][A-Z]+(?:\s+[A-Z][A-Z]+)*)')


def _clean_name_words(name: str) -> str:
    """Drop label words and trailing single-letter OCR artefacts from a name."""
    words = [word for word in name.split() if word.upper() not in NON_NAME_WORDS]
    # OCR picks up stray initials from the card border, e.g. "ZAKIR HOSSAIN D".
    while len(words) > 1 and len(words[-1]) == 1:
        words.pop()
    return ' '.join(words)


def extract_name(text: str) -> Optional[str]:
    """Extract name from text. Prioritize patterns that look like actual names."""
    # The "MD:" / "MD." honorific is the most specific signal available.
    match = MD_NAME_PATTERN.search(text)
    if match:
        separator = match.group(1)
        name = _clean_name_words(match.group(2))
        if name:
            return f"MD{separator} {name}"

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
            elif re.fullmatch(r'[A-Z]+', w) and w not in NON_NAME_WORDS:
                filtered.append(w)
        if filtered:
            return ' '.join(filtered)

    # Look for "Data of Birth" or "Date of Birth" context - name often appears before this
    dob_context_pattern = r'([A-Z][A-Z]+(?:\s+[A-Z]+)*)\s+(?:Data|Date)\s+of\s+Birth'
    match = re.search(dob_context_pattern, text)
    if match:
        name = match.group(1).strip()
        # Check if it looks like a reasonable name (not too long, not common words)
        if len(name.split()) <= 3 and not any(word in NON_NAME_WORDS for word in name.split()):
            return name

    # Fallback: try to find all-caps sequences, but filter out common non-name words
    fallback_pattern = r'(MD[\.,-]?\s*)?([A-Z]{2,}(?:\s+[A-Z]{2,})*)'
    all_caps = re.findall(fallback_pattern, text)
    if all_caps:
        # Filter out common non-name words and find the best match
        filtered_matches = []
        for prefix, name_part in all_caps:
            filtered_words = _clean_name_words(name_part.strip())
            if filtered_words:
                filtered_matches.append((prefix, filtered_words))

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
        bo_pattern = r'BO\s*Account\s*Number\s*[:\-]?\s*((?:\d\s*){16})'
        match = re.search(bo_pattern, text, re.IGNORECASE)
        # The trailing \s* in the pattern can absorb whitespace after the last digit.
        details["bo_id"] = match.group(1).strip() if match else None
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
    type: DocumentType = Form(..., description="Type of document: NID, BO, TIN"),
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
        # Extract fields based on type
        details = extract_fields_by_type(type.value, all_text)
        # Format response based on type
        if type == DocumentType.NID:
            response = {
                "type": type.value,
                "details": {
                    "name": details.get("name"),
                    "date_of_birth": details.get("date_of_birth"),
                    "nid_number": details.get("nid_number"),
                },
            }
        elif type == DocumentType.BO:
            response = {
                "type": type.value,
                "details": {
                    "bo_id": details.get("bo_id"),
                },
            }
        elif type == DocumentType.TIN:
            response = {
                "type": type.value,
                "details": {
                    "tin_number": details.get("tin_number"),
                },
            }
        else:
            response = {
                "type": type.value,
                "details": details,
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
