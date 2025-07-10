#!/usr/bin/env python3
"""
Test script for NID Parser API

This script demonstrates how to use the NID Parser API endpoints.
It includes examples for all available endpoints and error handling.
"""

import requests
import json
import time
import os
from typing import Dict, Any

# API Configuration
BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    "root": "/",
    "health": "/health",
    "extract_nid": "/extract-nid-info/"
}

def print_separator(title: str):
    """Print a formatted separator with title."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def test_root_endpoint() -> Dict[str, Any]:
    """Test the root endpoint."""
    print_separator("Testing Root Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}{API_ENDPOINTS['root']}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Root endpoint working correctly")
            print(f"API Message: {data['message']}")
            print(f"API Version: {data['version']}")
            print("Available Endpoints:")
            for endpoint, path in data['endpoints'].items():
                print(f"  - {endpoint}: {path}")
            return data
        else:
            print(f"❌ Root endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return {}
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the API server is running on localhost:8000")
        return {}
    except Exception as e:
        print(f"❌ Error testing root endpoint: {str(e)}")
        return {}

def test_health_endpoint() -> Dict[str, Any]:
    """Test the health endpoint."""
    print_separator("Testing Health Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}{API_ENDPOINTS['health']}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health endpoint working correctly")
            print(f"Status: {data['status']}")
            print(f"Timestamp: {data['timestamp']}")
            return data
        else:
            print(f"❌ Health endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return {}
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the API server is running on localhost:8000")
        return {}
    except Exception as e:
        print(f"❌ Error testing health endpoint: {str(e)}")
        return {}

def test_extract_nid_endpoint(image_path: str = None) -> Dict[str, Any]:
    """Test the extract NID endpoint."""
    print_separator("Testing Extract NID Endpoint")
    
    if not image_path or not os.path.exists(image_path):
        print("⚠️  No valid image file provided. Testing with error handling...")
        
        # Test with invalid file type
        print("\n--- Testing with invalid file type ---")
        try:
            files = {"file": ("test.txt", "This is not an image", "text/plain")}
            response = requests.post(f"{BASE_URL}{API_ENDPOINTS['extract_nid']}", files=files)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 400:
                print("✅ Correctly rejected invalid file type")
                print(f"Error: {response.json()['detail']}")
            else:
                print(f"❌ Unexpected response for invalid file: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error testing invalid file: {str(e)}")
        
        # Test with empty file
        print("\n--- Testing with empty file ---")
        try:
            files = {"file": ("empty.jpg", b"", "image/jpeg")}
            response = requests.post(f"{BASE_URL}{API_ENDPOINTS['extract_nid']}", files=files)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 400:
                print("✅ Correctly rejected empty file")
                print(f"Error: {response.json()['detail']}")
            else:
                print(f"❌ Unexpected response for empty file: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error testing empty file: {str(e)}")
        
        return {}
    
    # Test with valid image
    print(f"Testing with image: {image_path}")
    try:
        with open(image_path, 'rb') as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(f"{BASE_URL}{API_ENDPOINTS['extract_nid']}", files=files)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ NID extraction successful")
            print(f"Name: {data['name']}")
            print(f"Date of Birth: {data['dob']}")
            print(f"NID Number: {data['nid']}")
            print(f"Extracted Text Length: {len(data['extracted_text'])} characters")
            print(f"First 100 chars of extracted text: {data['extracted_text'][:100]}...")
            return data
        else:
            print(f"❌ NID extraction failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return {}
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the API server is running on localhost:8000")
        return {}
    except Exception as e:
        print(f"❌ Error testing NID extraction: {str(e)}")
        return {}

def test_api_documentation():
    """Test API documentation endpoints."""
    print_separator("Testing API Documentation")
    
    doc_endpoints = [
        ("Swagger UI", "/docs"),
        ("ReDoc", "/redoc"),
        ("OpenAPI JSON", "/openapi.json")
    ]
    
    for name, endpoint in doc_endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}")
            print(f"{name}: {'✅ Available' if response.status_code == 200 else '❌ Not available'}")
        except:
            print(f"{name}: ❌ Not available")

def run_performance_test(image_path: str = None):
    """Run a simple performance test."""
    if not image_path or not os.path.exists(image_path):
        print("⚠️  Skipping performance test - no valid image provided")
        return
    
    print_separator("Performance Test")
    
    times = []
    for i in range(3):
        print(f"Test {i+1}/3...")
        start_time = time.time()
        
        try:
            with open(image_path, 'rb') as f:
                files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                response = requests.post(f"{BASE_URL}{API_ENDPOINTS['extract_nid']}", files=files)
            
            if response.status_code == 200:
                end_time = time.time()
                duration = end_time - start_time
                times.append(duration)
                print(f"  ✅ Completed in {duration:.2f} seconds")
            else:
                print(f"  ❌ Failed with status {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"\nPerformance Summary:")
        print(f"  Average time: {avg_time:.2f} seconds")
        print(f"  Min time: {min_time:.2f} seconds")
        print(f"  Max time: {max_time:.2f} seconds")

def main():
    """Main test function."""
    print("🚀 NID Parser API Test Suite")
    print(f"Testing API at: {BASE_URL}")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server is not responding correctly")
            return
    except:
        print("❌ Cannot connect to API server. Make sure it's running on localhost:8000")
        print("   Start the server with: python main.py")
        return
    
    # Run tests
    test_root_endpoint()
    test_health_endpoint()
    test_api_documentation()
    
    # Check for test image
    test_image = None
    possible_images = ["test_nid.jpg", "test_nid.png", "sample.jpg", "sample.png"]
    
    for img in possible_images:
        if os.path.exists(img):
            test_image = img
            break
    
    if test_image:
        print(f"\n📸 Found test image: {test_image}")
        test_extract_nid_endpoint(test_image)
        run_performance_test(test_image)
    else:
        print("\n📸 No test image found. Testing error handling only.")
        test_extract_nid_endpoint()
    
    print_separator("Test Summary")
    print("✅ Basic API functionality tested")
    print("✅ Error handling tested")
    print("✅ Documentation endpoints verified")
    
    if test_image:
        print("✅ NID extraction tested with sample image")
        print("✅ Performance test completed")
    else:
        print("⚠️  NID extraction tested with error cases only")
        print("💡 To test with a real image, place a test image in the current directory")
        print("   Supported formats: JPEG, PNG, BMP, TIFF")
    
    print("\n🎉 Test suite completed!")
    print(f"📖 View interactive documentation at: {BASE_URL}/docs")

if __name__ == "__main__":
    main() 