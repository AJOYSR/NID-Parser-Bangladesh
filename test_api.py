#!/usr/bin/env python3
"""
Test script for the NID Parser API
This script demonstrates how to use the API endpoints
"""

import requests
import json
import os
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running.")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_root_endpoint():
    """Test the root endpoint."""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Root endpoint working")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_extract_nid_info(image_path):
    """Test the NID extraction endpoint with an image file."""
    print(f"\nTesting NID extraction with image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/extract-nid-info/", files=files)
        
        if response.status_code == 200:
            print("✅ NID extraction successful")
            result = response.json()
            print("Extracted Information:")
            print(f"  Name: {result.get('name', 'Not detected')}")
            print(f"  Date of Birth: {result.get('dob', 'Not detected')}")
            print(f"  NID Number: {result.get('nid', 'Not detected')}")
        else:
            print(f"❌ NID extraction failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_invalid_file():
    """Test the API with an invalid file type."""
    print("\nTesting with invalid file type...")
    try:
        # Create a temporary text file
        with open("test.txt", "w") as f:
            f.write("This is not an image file")
        
        with open("test.txt", "rb") as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/extract-nid-info/", files=files)
        
        if response.status_code == 400:
            print("✅ Correctly rejected invalid file type")
        else:
            print(f"❌ Should have rejected invalid file: {response.status_code}")
        
        # Clean up
        os.remove("test.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all tests."""
    print("🚀 Starting NID Parser API Tests")
    print("=" * 50)
    
    # Test basic endpoints
    test_health_check()
    test_root_endpoint()
    
    # Test invalid file
    test_invalid_file()
    
    # Test with actual image (if available)
    # You can replace this path with your actual NID image
    test_images = [
        "sample_nid.jpg",
        "test_image.png", 
        "nid_sample.jpg"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            test_extract_nid_info(image_path)
            break
    else:
        print("\n📝 No test images found. To test with an image:")
        print("1. Place a NID image in the project directory")
        print("2. Update the image_path in this script")
        print("3. Run the test again")
    
    print("\n" + "=" * 50)
    print("🏁 Tests completed!")

if __name__ == "__main__":
    main() 