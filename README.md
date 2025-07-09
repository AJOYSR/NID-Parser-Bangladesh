# NID Parser API

A FastAPI-based OCR service that extracts structured information from National ID (NID) images using EasyOCR.

## Features

- **Image Upload**: Accept image files via HTTP POST requests
- **OCR Processing**: Uses EasyOCR for text extraction from images
- **Structured Output**: Returns extracted information in JSON format:
  - `name`: Detected person's name
  - `dob`: Detected date of birth
  - `nid`: Detected NID number
- **Multiple Date Formats**: Supports various date formats (DD/MM/YYYY, YYYY/MM/DD, etc.)
- **Flexible NID Detection**: Handles different NID number formats
- **Name Extraction**: Intelligent name detection from OCR text

## Setup Instructions

### 1. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\\Scripts\\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API

```bash
# Run with uvicorn (recommended for development)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Or run directly with Python (uses default port 8000)
python main.py
```

The API will be available at `http://localhost:8000`

## Logs and Model Storage

- **Logs**: Log files and output are stored in the `logs/` directory (ignored by git).
- **EasyOCR Models**: Downloaded models are stored in the `models/` directory (ignored by git).

## .gitignore Coverage

The repository's `.gitignore` is set up to exclude:
- Python virtual environments (`venv/`, `env/`, etc.)
- Compiled Python files and caches (`__pycache__/`, `*.pyc`, etc.)
- Log files and directories (`logs/`, `*.log`)
- Temporary files and folders (`*.tmp`, `temp/`, etc.)
- EasyOCR model files (`*.pth`, `*.onnx`)
- IDE/editor settings (`.vscode/`, `.idea/`)

## API Endpoints

### 1. Extract NID Information
**POST** `/extract-nid-info/`

Upload an image file to extract structured NID information.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
```json
{
  "name": "John Doe",
  "dob": "15/03/1990",
  "nid": "123456789012345"
}
```

### 2. Health Check
**GET** `/health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00"
}
```

### 3. API Documentation
**GET** `/docs`

Interactive API documentation (Swagger UI)

## Usage Examples

### Using curl

```bash
# Upload an image and extract NID info
curl -X POST "http://localhost:8000/extract-nid-info/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/nid_image.jpg"
```

### Using Python requests

```python
import requests

url = "http://localhost:8000/extract-nid-info/"
files = {"file": open("nid_image.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Name: {result['name']}")
print(f"Date of Birth: {result['dob']}")
print(f"NID Number: {result['nid']}")
```

## Configuration

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

### OCR Settings
- **Language**: English (en)
- **GPU**: Disabled by default (set `gpu=True` in `main.py` if you have GPU support)
- **Detail Level**: Full detail with bounding boxes

### Customization

You can modify the extraction patterns in `main.py`:

1. **Date Patterns**: Update `extract_date_of_birth()` function
2. **NID Patterns**: Update `extract_nid_number()` function  
3. **Name Patterns**: Update `extract_name()` function

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid file type or missing file
- **500 Internal Server Error**: OCR processing failures
- **Detailed Logging**: All operations are logged for debugging

## Performance Notes

- First request may be slower due to EasyOCR model loading
- Subsequent requests will be faster with cached reader instance
- Large images may take longer to process
- Consider image preprocessing for better OCR accuracy

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Use smaller images or reduce image quality
3. **Slow Processing**: Consider using GPU if available
4. **Poor OCR Results**: Ensure images are clear and well-lit
5. **macOS SSL or PIL Errors**: The code includes fixes for SSL certificate and PIL.Image.ANTIALIAS compatibility issues on macOS. If you encounter related errors, ensure you are using the provided code and have the latest Pillow installed.

### Logs

Check the console output for detailed logs and error messages. Log files are stored in the `logs/` directory (if configured).

## License

This project is open source and available under the MIT License. 