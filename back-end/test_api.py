import requests
import base64
import numpy as np
from PIL import Image
import io
import json

# API endpoint
API_URL = "http://localhost:8000/predict"

def test_with_sample_image():
    """Test with a simple hand-drawn digit image"""
    print("🎨 Testing with sample image...")
    
    # Create a simple test image (a basic "1" shape)
    img_array = np.zeros((28, 28), dtype=np.uint8)
    
    # Draw a simple "1" in the middle
    for i in range(8, 20):
        img_array[i, 14] = 255  # Vertical line
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    #view the image
    img.show()
    
    # Test the API
    test_api_call(img_str, "Sample '1' image")

def test_with_mnist_sample():
    """Test with an actual MNIST digit"""
    print("📊 Testing with MNIST sample...")
    
    try:
        from tensorflow import keras
        from keras.datasets import mnist
        (_, _), (test_images, test_labels) = mnist.load_data()
        
        # Take the first test image
        sample_image = test_images[0]
        true_label = test_labels[0]
        
        # Convert to PIL Image
        img = Image.fromarray(sample_image)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        print(f"MNIST sample - True label: {true_label}")
        test_api_call(img_str, f"MNIST digit {true_label}")
        
    except ImportError:
        print("TensorFlow not available, skipping MNIST test")

def test_api_call(img_base64, description):
    """Make API call and display results"""
    try:
        # Prepare the request
        payload = {"data_url": f"data:image/png;base64,{img_base64}"}
        
        # Make the request
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if it's an error response or success response
            if "error" in result:
                print(f"⚠️  {description} - API returned error:")
                print(f"   Error: {result['error']}")
            elif "digit" in result:
                print(f"✅ {description} - Success!")
                print(f"   Predicted digit: {result['digit']}")
                print(f"   Probabilities: {[f'{p:.3f}' for p in result['probs']]}")
                print(f"   Confidence: {result['probs'][result['digit']]:.3f}")
            else:
                print(f"❓ {description} - Unexpected response format:")
                print(f"   Response: {result}")
        else:
            print(f"❌ {description} - HTTP Error!")
            print(f"   Status code: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ {description} - Connection failed!")
        print("   Make sure your server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ {description} - Error: {str(e)}")

def test_error_cases():
    """Test various error scenarios"""
    print("\n🚨 Testing error cases...")
    
    # Test with invalid base64
    print("Testing invalid base64...")
    test_api_call("invalid_base64_data", "Invalid base64")
    
    # Test with empty data
    print("Testing empty data...")
    try:
        response = requests.post(API_URL, json={})
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {str(e)}")

def main():
    print("🚀 Starting API Tests...")
    print("=" * 50)
    
    # Test 1: Sample image
    test_with_sample_image()
    print()
    
    # Test 2: MNIST sample (if available)
    test_with_mnist_sample()
    print()
    
    # Test 3: Error cases
    test_error_cases()
    
    print("\n" + "=" * 50)
    print("✨ Testing complete!")

if __name__ == "__main__":
    main()
