document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const signItems = document.querySelectorAll('.sign-item');
    const modal = document.getElementById('webcamModal');
    const closeBtn = document.querySelector('.close-btn');
    const currentSignElement = document.getElementById('currentSign');
    const targetSignElement = document.getElementById('target-sign');
    const scoreDisplayElement = document.getElementById('score-display');
    const webcam = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const overlay = document.getElementById('overlay');
    const startBtn = document.getElementById('startBtn');
    const practiceBtn = document.getElementById('practiceBtn');
    const instructions = document.getElementById('instructions');

    // Variables
    let currentSign = '';
    let stream = null;
    let isCapturing = false;
    let captureInterval = null;
    const canvasContext = canvas.getContext('2d');
    const overlayContext = document.createElement('canvas').getContext('2d');

    // Event Listeners
    signItems.forEach(item => {
        item.addEventListener('click', () => {
            currentSign = item.getAttribute('data-sign');
            openModal(currentSign);
        });
    });

    closeBtn.addEventListener('click', closeModal);
    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });

    startBtn.addEventListener('click', toggleCamera);
    practiceBtn.addEventListener('click', togglePractice);

    // Functions
    function openModal(sign) {
        currentSignElement.textContent = sign;
        targetSignElement.textContent = sign;
        modal.style.display = 'block';
        scoreDisplayElement.textContent = '';
        
        // Reset states
        if (isCapturing) {
            stopCapture();
        }
        if (stream) {
            stopCamera();
            startBtn.textContent = 'Start Camera';
        }
        
        practiceBtn.disabled = true;
    }

    function closeModal() {
        modal.style.display = 'none';
        if (isCapturing) {
            stopCapture();
        }
        if (stream) {
            stopCamera();
        }
    }

    async function toggleCamera() {
        if (stream) {
            stopCamera();
            startBtn.textContent = 'Start Camera';
            practiceBtn.disabled = true;
        } else {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'user' }
                });
                webcam.srcObject = stream;
                startBtn.textContent = 'Stop Camera';
                practiceBtn.disabled = false;
                
                // Set canvas dimensions based on video
                webcam.onloadedmetadata = () => {
                    canvas.width = webcam.videoWidth;
                    canvas.height = webcam.videoHeight;
                    overlay.style.width = `${webcam.videoWidth}px`;
                    overlay.style.height = `${webcam.videoHeight}px`;
                };
            } catch (err) {
                console.error('Error accessing webcam:', err);
                instructions.textContent = 'Error: Could not access webcam. Please check your camera permissions.';
            }
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcam.srcObject = null;
            stream = null;
        }
    }

    function togglePractice() {
        if (isCapturing) {
            stopCapture();
            practiceBtn.textContent = 'Start Practice';
            instructions.textContent = 'Click "Start Practice" to begin recognizing signs.';
        } else {
            startCapture();
            practiceBtn.textContent = 'Stop Practice';
            instructions.textContent = 'Make the sign for "' + currentSign + '" and hold it steady for recognition.';
        }
    }

    function startCapture() {
        isCapturing = true;
        captureInterval = setInterval(captureFrame, 200); // Capture frame every 200ms for more responsive feedback
    }

    function stopCapture() {
        isCapturing = false;
        clearInterval(captureInterval);
        // Clear overlay
        overlay.innerHTML = '';
    }

    function captureFrame() {
        if (!isCapturing || !stream) return;
        
        // Draw current video frame to canvas
        canvasContext.drawImage(webcam, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to data URL
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to server for processing
        processFrame(imageData);
    }

    async function processFrame(imageData) {
        try {
            const response = await fetch('/learn/process_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: imageData,
                    target_sign: currentSign
                })
            });
            
            const result = await response.json();
            
            // Update UI with recognition results
            updateUI(result);
            
        } catch (error) {
            console.error('Error processing frame:', error);
        }
    }

    function updateUI(result) {
        // Clear previous overlay content
        overlay.innerHTML = '';
        
        // Update score display
        scoreDisplayElement.textContent = `No match`;
        
        // Highlight if it's a match
        if (result.match) {
            scoreDisplayElement.style.backgroundColor = 'rgba(0, 128, 0, 0.7)'; // Green for match
            scoreDisplayElement.textContent = `Score: ${result.score}%`;
        } else {
            scoreDisplayElement.style.backgroundColor = 'rgba(0, 0, 0, 0.7)'; // Default black
        }
        
        // // Draw bounding boxes for hands
        // result.hand_boxes.forEach(box => {
        //     const [x_min, y_min, x_max, y_max] = box;
            
        //     // Create div for bounding box
        //     const boundingBox = document.createElement('div');
        //     boundingBox.style.position = 'absolute';
        //     boundingBox.style.left = `${(x_min / canvas.width) * 50}%`;
        //     boundingBox.style.top = `${(y_min / canvas.height) * 50}%`;
        //     boundingBox.style.width = `${((x_max - x_min) / canvas.width) * 100}%`;
        //     boundingBox.style.height = `${((y_max - y_min) / canvas.height) * 100}%`;
        //     boundingBox.style.border = '2px solid #00ff00';
        //     boundingBox.style.boxSizing = 'border-box';
            
        //     overlay.appendChild(boundingBox);
        // });
    }
});