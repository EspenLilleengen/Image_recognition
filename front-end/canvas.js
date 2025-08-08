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
    
    // Drawing variables
    let painting = false;
    let brushSize = 5;
    let brushColor = "#ffffff";
    
    // Initialize canvas size
    function initializeCanvas() {
        canvas.width = parseInt(canvasWidthInput.value);
        canvas.height = parseInt(canvasHeightInput.value);
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    
    // Resize canvas function
    function resizeCanvas() {
        canvas.width = parseInt(canvasWidthInput.value);
        canvas.height = parseInt(canvasHeightInput.value);
        ctx.fillStyle = "black";
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
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    
    // Download canvas function
    function downloadCanvas() {
        const link = document.createElement('a');
        link.download = 'drawing.png';
        link.href = canvas.toDataURL();
        link.click();
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