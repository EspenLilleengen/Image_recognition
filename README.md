# Image Recognition - MNIST Digit Classifier

Check out the application [here](https://espenlilleengen.github.io/Image_recognition/).
Front-end is hosted on Github pages, while back-end is hosted on Render. Since Back-end is hosted on a free version of Render it mat take some time (up to 50 seconds) to get a classification after sending the first request, the first classification might result in an error. 

A web-based MNIST digit classifier that allows users to draw digits on a canvas and get real-time predictions using a trained neural network. This project is my first full-stack application and is still a work-in-progress.


## 🏗️ Project Structure

```
Image_recognition/
├── back-end/                 # FastAPI backend server
│   ├── main.py              # FastAPI application with prediction endpoint
│   ├── run.py               # Server startup script
│   ├── training.py          # Model training script
│   ├── requirements.txt     # Python dependencies
│   ├── best_model.keras    # Pre-trained model (best performing)
│   ├── my_model.keras      # Alternative trained model
│   ├── test_api.py         # API testing script
│   └── data/               # MNIST dataset storage
│       └── MNIST/
│           └── raw/        # Raw MNIST data files
├── front-end/               # Web interface
│   ├── index.html          # Main HTML page
│   ├── style.css           # Styling and layout
│   └── canvas.js           # Canvas drawing and API interaction
└── README.md               # This file
```

## 🚀 Features

- **Interactive Drawing Canvas**: Draw digits with customizable brush size and colors
- **Real-time Classification**: Instant digit prediction using trained neural network
- **Probability Visualization**: See confidence scores for all possible digits (0-9)
- **Responsive Design**: Works on desktop and mobile devices
- **Model Training**: Complete training pipeline with data augmentation
- **RESTful API**: FastAPI backend with CORS support

## 🛠️ Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Image_recognition
```

### 2. Install Backend Dependencies
```bash
cd back-end
pip install -r requirements.txt
```

### 3. Download MNIST Dataset (Optional)
The training script will automatically download the MNIST dataset on first run, but you can also manually download it:
```bash
cd data/MNIST/raw
# The training script will handle this automatically
```

## 🚀 Usage

### Starting the Backend Server

1. **Navigate to the backend directory:**
   ```bash
   cd back-end
   ```

2. **Start the FastAPI server:**
   ```bash
   python run.py
   ```
   
   The server will start on `http://localhost:8000`

3. **Alternative startup method:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Using the Web Interface
-Uncomment the localhost link in config.js

1. **Open the frontend:**
   - Navigate to `http://127.0.0.1:5500/front-end/index.html` in your web browser
   - Or serve it using a local web server

2. **Drawing Interface:**
   - Use the canvas to draw digits (0-9)

3. **Classification:**
   - Click "Classify" to get predictions
   - View the predicted digit and confidence scores
   - Clear canvas to draw new digits

### Training Your Own Model

1. **Navigate to the backend directory:**
   ```bash
   cd back-end
   ```

2. **Run the training script:**
   ```bash
   python training.py
   ```

3. **Training features:**
   - Data augmentation (rotation, shift, zoom)
   - Early stopping to prevent overfitting
   - Learning rate scheduling
   - Model checkpointing


## 🧠 Model Architecture

The neural network uses a CNN architecture optimized for MNIST digit classification:

- **Input Layer**: 28x28x1 grayscale images
- **Convolutional Blocks**: 2 blocks with batch normalization and dropout
- **Pooling Layers**: MaxPooling2D for dimension reduction
- **Dense Layers**: Fully connected layers with dropout
- **Output Layer**: 10 classes (digits 0-9) with softmax activation

## 📊 Performance

- **Training Accuracy**: ~99.5%
- **Validation Accuracy**: ~99.2%
- **Test Accuracy**: ~99.1%
- **Inference Time**: <100ms per prediction

## 🧪 Testing

### Test the API
```bash
cd back-end
python test_api.py
```

### Manual Testing
1. Start the backend server
2. Open the frontend in a browser
3. Draw digits and verify predictions
4. Test with various brush sizes and colors

## Troubleshooting

- **"Error classifying image"**: Make sure the backend server is running on port 8000
- **Canvas not working**: Check browser console for JavaScript errors
- **Model not loading**: Ensure `my_model.keras` exists in the back-end directory

## Future Enhancements

- Website accesible application
- Real-time prediction as you draw
- Support for different drawing styles
- Model retraining interface
- Export/import functionality for drawings
- Multi-digit recognition
- More confident percentages 


