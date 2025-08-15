import config from "./config.js";

window.addEventListener("load", () => {
    const canvas = document.querySelector("#canvas");
    const ctx = canvas.getContext("2d");
    
    // Get control elements
    const canvasWidthInput = document.querySelector("#canvas-width");
    const canvasHeightInput = document.querySelector("#canvas-height");
    const resizeBtn = document.querySelector("#resize-btn");
    const brushSizeInput = document.querySelector("#brush-size");
    const brushSizeValue = document.querySelector("#brush-size-value");
    const colorPicker = document.querySelector("#color-picker");
    const clearBtn = document.querySelector("#clear-btn");
    const downloadBtn = document.querySelector("#download-btn");
    const classifyBtn = document.querySelector("#classify-btn");
    
    // Drawing variables
    let painting = false;
    let brushSize = brushSizeValue.textContent;
    let brushColor = colorPicker.value;
    const backgroundColor = "#000000"; // Always black

    // Initialize canvas size
    function initializeCanvas() {
        canvas.width = parseInt(canvasWidthInput.value);
        canvas.height = parseInt(canvasHeightInput.value);
        ctx.fillStyle = backgroundColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    
    // Resize canvas function
    function resizeCanvas() {
        canvas.width = parseInt(canvasWidthInput.value);
        canvas.height = parseInt(canvasHeightInput.value)
        ctx.fillStyle = backgroundColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    
    // Get mouse position relative to canvas
    function getMousePos(canvas, event) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
    }
    
    // Drawing functions
    function startPosition(event) {
        painting = true;
        const pos = getMousePos(canvas, event);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    }
    
    function finishedPosition() {
        painting = false;
        ctx.beginPath();
    }
    
    function draw(event) {
        if (!painting) return;
        
        const pos = getMousePos(canvas, event);
        
        ctx.lineWidth = brushSize;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = brushColor;
        
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }
    
    // Clear canvas function
    function clearCanvas() {
        ctx.fillStyle = backgroundColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Clear classification results
        clearClassificationResults();
    }
    // Clear classification results function
    function clearClassificationResults() {
        const predictedDigit = document.querySelector("#predicted-digit");
        const probBars = document.querySelector("#prob-bars");
        
        predictedDigit.textContent = "Draw a digit and click Classify";
        probBars.innerHTML = "";
    }
    
    // Download canvas function
    function downloadCanvas() {
        const link = document.createElement('a');
        link.download = 'drawing.png';
        link.href = canvas.toDataURL();
        link.click();
    }
    
    // Classify canvas function
    async function classifyCanvas() {
        try {
            // Show loading state
            classifyBtn.textContent = "Classifying...";
            classifyBtn.disabled = true;
            
            // Get canvas data as base64
            const dataUrl = canvas.toDataURL();
            
            // Send to backend
            const response = await fetch(`${config.API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ data_url: dataUrl })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            // Display result
            displayClassificationResult(result);
            
        } catch (error) {
            console.error('Classification error:', error);
            alert(`Classification failed: ${error.message}`);
        } finally {
            // Reset button state
            classifyBtn.textContent = "Classify";
            classifyBtn.disabled = false;
        }
    }
    
    // Display classification result
    function displayClassificationResult(result) {
        // Update existing result display elements
        const predictedDigit = document.querySelector("#predicted-digit");
        const probBars = document.querySelector("#prob-bars");
        
        // Update predicted digit
        predictedDigit.innerHTML = `<strong>Predicted Digit:</strong> ${result.digit}`;
        
        // Update probability bars
        probBars.innerHTML = result.probs.map((prob, index) => `
            <div class="prob-item">
                <span class="digit">${index}</span>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: ${(prob * 100).toFixed(1)}%"></div>
                </div>
                <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    }
    
    // Event listeners for drawing
    canvas.addEventListener("mousedown", startPosition);
    canvas.addEventListener("mouseup", finishedPosition);
    canvas.addEventListener("mouseout", finishedPosition);
    canvas.addEventListener("mousemove", draw);
    
    // Event listeners for controls
    resizeBtn.addEventListener("click", resizeCanvas);
    clearBtn.addEventListener("click", clearCanvas);
    downloadBtn.addEventListener("click", downloadCanvas);
    classifyBtn.addEventListener("click", classifyCanvas);
    
    // Update brush size display and value
    brushSizeInput.addEventListener("input", () => {
        brushSize = parseInt(brushSizeInput.value);
        brushSizeValue.textContent = brushSize;
    });
    
    // Update brush color
    colorPicker.addEventListener("input", () => {
        brushColor = colorPicker.value;
    });
    
    // Initialize canvas on load
    initializeCanvas();
});