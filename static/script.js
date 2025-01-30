// Webcam Setup
let webcamStream;

async function startWebcam() {
    try {
        const video = document.getElementById('webcam');
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = webcamStream;
    } catch (err) {
        console.error('Error accessing webcam:', err);
    }
}

// Image Capture and Processing
async function captureImage() {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg');
    const result = await processImage(imageData);
    document.getElementById('signResult').innerText = `Sign Detected: ${result}`;
}

// Model Interaction Functions
async function processImage(imageData) {
    // Send to PyTorch backend
    const response = await fetch('/analyze-cnn', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData })
    });
    return await response.json();
}

async function analyzeEmotion() {
    const text = document.getElementById('userInput').value;
    const response = await fetch('/analyze-emotion', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    });
    const result = await response.json();
    document.getElementById('analysisResult').innerText = 
        `Emotion: ${result.emotion} (${Math.round(result.confidence * 100)}%)`;
}

function formatIntentName(intent) {
    return intent.split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

async function analyzeIntent() {
    const text = document.getElementById('intentInput').value;
    const response = await fetch('/analyze-intent', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    });
    const result = await response.json();
    
    // Create formatted HTML for results
    const resultHTML = `
        <div class="intent-results">
            <div class="input-text">
                <strong>Input:</strong> "${result.input_text}"
            </div>
            <div class="predictions">
                <strong>Top Predictions:</strong>
                ${result.top_intents.map((intent, index) => `
                    <div class="prediction ${index === 0 ? 'top-prediction' : ''}">
                        <div class="intent-name">${formatIntentName(intent.intent)}</div>
                        <div class="confidence-bar">
                            <div class="bar" style="width: ${intent.confidence * 100}%"></div>
                            <span>${Math.round(intent.confidence * 100)}%</span>
                        </div>
                        <div class="description">${intent.description}</div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    document.getElementById('intentAnalysisResult').innerHTML = resultHTML;
}

// File Upload Handler
document.getElementById('imageUpload').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const reader = new FileReader();
    
    reader.onload = async function(event) {
        const result = await processImage(event.target.result);
        document.getElementById('signResult').innerText = `Sign Detected: ${result}`;
    };
    
    reader.readAsDataURL(file);
});