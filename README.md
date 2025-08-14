# Image Recognition - MNIST Digit Classifier

This project is a web-based MNIST digit classifier that allows users to draw digits on a canvas and get real-time predictions using a trained neural network.

## Features

- Interactive drawing canvas with customizable brush size and color
- Real-time digit classification using a pre-trained Keras model
- Beautiful visualization of prediction probabilities
- Responsive design for both desktop and mobile devices

## Project Structure

```
Image_recognition/
├── back-end/           # FastAPI backend with ML model
│   ├── main.py        # API server and prediction logic
│   ├── my_model.keras # Trained MNIST model
│   └── training.py    # Model training script
├── front-end/         # Web interface
│   ├── index.html     # Main HTML page
│   ├── canvas.js      # Drawing and classification logic
│   └── style.css      # Styling and layout
└── README.md          # This file
```

## Setup and Usage

### 1. Install Dependencies

```bash
# Backend dependencies
pip install fastapi uvicorn tensorflow pillow numpy

# Or install from requirements.txt if available
pip install -r requirements.txt
```

### 2. Start the Backend Server

```bash
cd back-end
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 3. Open the Frontend

Open `front-end/index.html` in your web browser, or serve it using a local server:

```bash
cd front-end
python -m http.server 8080
# Then open http://localhost:8080
```

### 4. Use the Application

1. **Draw a digit**: Use your mouse or touch to draw a digit (0-9) on the canvas
2. **Adjust settings**: Change brush size, color, or canvas dimensions as needed
3. **Classify**: Click the "Classify" button to send your drawing to the ML model
4. **View results**: See the predicted digit and confidence scores for all possibilities

## API Endpoints

- `POST /predict` - Accepts a base64-encoded image and returns digit prediction
  - Input: `{"data_url": "data:image/png;base64,..."}`
  - Output: `{"digit": 5, "probs": [0.01, 0.02, 0.03, 0.04, 0.05, 0.85, ...]}`

## How It Works

1. **Frontend**: The canvas captures user drawings and converts them to base64 data URLs
2. **Backend**: Receives the image data, preprocesses it (resize to 28x28, convert to grayscale, normalize)
3. **ML Model**: The trained Keras model predicts the digit and returns probability scores
4. **Results**: Frontend displays the prediction with a visual probability bar chart

## Technical Details

- **Model**: Pre-trained Keras model trained on MNIST dataset
- **Input**: 28x28 grayscale images (normalized to 0-1 range)
- **Output**: 10-class classification (digits 0-9) with probability scores
- **Framework**: FastAPI backend, vanilla JavaScript frontend
- **CORS**: Enabled for cross-origin requests

## Troubleshooting

- **"Error classifying image"**: Make sure the backend server is running on port 8000
- **Canvas not working**: Check browser console for JavaScript errors
- **Model not loading**: Ensure `my_model.keras` exists in the back-end directory

## Future Enhancements

- Real-time prediction as you draw
- Support for different drawing styles
- Model retraining interface
- Export/import functionality for drawings
- Multi-digit recognition 
