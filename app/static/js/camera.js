document.addEventListener('DOMContentLoaded', () => {
    const cameraView = document.getElementById('camera-view');
    const cameraCanvas = document.getElementById('camera-canvas');
    const guidesCanvas = document.getElementById('guides-canvas');
    const cameraFeedback = document.getElementById('camera-feedback');
    const startCameraBtn = document.getElementById('start-camera');
    const captureBtn = document.getElementById('capture-btn');
    const switchCameraBtn = document.getElementById('switch-camera');
    const flash = document.getElementById('flash');
    const capturedImagesContainer = document.getElementById('captured-images');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    // Metrics elements
    const lightingProgress = document.getElementById('lighting-progress');
    const angleProgress = document.getElementById('angle-progress');
    const iqiProgress = document.getElementById('iqi-progress');
    const lightingValue = document.getElementById('lighting-value');
    const angleValue = document.getElementById('angle-value');
    const iqiValue = document.getElementById('iqi-value');
    const statusMessage = document.getElementById('status-message');
    
    // Camera settings
    let stream = null;
    let currentCameraIndex = 0;
    let availableCameras = [];
    let processingFrame = false;
    let animationFrameId = null;
    let latestMetrics = null;
    let isFrontCamera = false; // Track which camera is being used
    
    // Constants
    const QUALITY_THRESHOLD = 70;
    const ANALYSIS_INTERVAL = 500; // Analyze frames every 500ms
    let lastAnalysisTime = 0;
    let errorCount = 0;
    let networkStatus = true;
    
    // Function to check server connection
    async function checkServerConnection() {
        try {
            const response = await fetch('/', { 
                method: 'HEAD',
                cache: 'no-store'
            });
            
            if (response.ok && !networkStatus) {
                // Connection restored
                networkStatus = true;
                updateStatus('Connection restored', 'success');
                setTimeout(() => {
                    if (networkStatus) {
                        updateStatus('Camera ready', 'success');
                    }
                }, 2000);
                errorCount = 0;
            }
            return response.ok;
        } catch (error) {
            if (networkStatus) {
                networkStatus = false;
                updateStatus('Connection to server lost. Please check your network.', 'error');
            }
            return false;
        }
    }
    
    // Get the list of available cameras
    async function getAvailableCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(device => device.kind === 'videoinput');
        } catch (error) {
            console.error('Error enumerating devices:', error);
            return [];
        }
    }
    
    // Initialize camera
    async function initCamera() {
        try {
            availableCameras = await getAvailableCameras();
            
            if (availableCameras.length === 0) {
                updateStatus('No cameras found', 'error');
                return false;
            }
            
            switchCameraBtn.disabled = availableCameras.length <= 1;
            
            return true;
        } catch (error) {
            console.error('Error initializing camera:', error);
            updateStatus('Failed to initialize camera', 'error');
            return false;
        }
    }
    
    // Start camera with selected device
    async function startCamera() {
        try {
            if (stream) {
                stopCamera();
            }
            
            // Show loading indicator
            loadingIndicator.classList.add('active');
            
            // Check if running on mobile
            const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
            
            // If on mobile and we have a specific camera index, try to use it
            let constraints = {};
            
            if (isMobile) {
                // On mobile, explicitly request front or back camera
                constraints = {
                    video: {
                        facingMode: isFrontCamera ? "user" : "environment",
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    }
                };
                
                // Update button text to reflect current camera
                switchCameraBtn.textContent = isFrontCamera ? "Use Back Camera" : "Use Front Camera";
                switchCameraBtn.disabled = false; // Always enable on mobile
            } else {
                // On desktop, use deviceId if available
                const deviceId = availableCameras.length > 0 ? 
                    availableCameras[currentCameraIndex].deviceId : undefined;
                
                constraints = {
                    video: {
                        deviceId: deviceId ? {exact: deviceId} : undefined,
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    }
                };
            }
            
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            cameraView.srcObject = stream;
            
            // Set canvas size once video metadata is loaded
            cameraView.onloadedmetadata = () => {
                // Hide loading indicator
                loadingIndicator.classList.remove('active');
                
                cameraCanvas.width = cameraView.videoWidth;
                cameraCanvas.height = cameraView.videoHeight;
                guidesCanvas.width = cameraView.videoWidth;
                guidesCanvas.height = cameraView.videoHeight;
                
                drawGuides();
                startFrameProcessing();
            };
            
            captureBtn.disabled = false;
            startCameraBtn.textContent = 'Restart Camera';
            updateStatus('Camera started. Position the chiller within the guides.');
            
            return true;
        } catch (error) {
            // Hide loading indicator
            loadingIndicator.classList.remove('active');
            
            console.error('Error starting camera:', error);
            if (error.name === 'NotAllowedError') {
                updateStatus('Camera access denied. Please allow camera access and try again.', 'error');
            } else if (error.name === 'NotFoundError') {
                updateStatus('No camera found. Please connect a camera and try again.', 'error');
            } else if (error.name === 'NotReadableError' || error.name === 'AbortError') {
                updateStatus('Camera is already in use by another application.', 'error');
            } else {
                updateStatus('Failed to start camera: ' + error.message, 'error');
            }
            return false;
        }
    }
    
    // Stop the camera
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        
        cameraView.srcObject = null;
        captureBtn.disabled = true;
    }
    
    // Switch to next available camera
    async function switchCamera() {
        // Check if running on mobile
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        
        if (isMobile) {
            // On mobile, toggle between front and back cameras
            isFrontCamera = !isFrontCamera;
            await startCamera();
        } else {
            // On desktop, cycle through available cameras
            if (availableCameras.length <= 1) return;
            
            currentCameraIndex = (currentCameraIndex + 1) % availableCameras.length;
            await startCamera();
        }
    }
    
    // Draw guidelines for positioning the chiller
    function drawGuides() {
        const ctx = guidesCanvas.getContext('2d');
        const width = guidesCanvas.width;
        const height = guidesCanvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        // Set line style
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        
        // Draw center crosshair
        ctx.beginPath();
        ctx.moveTo(width/2, height/2 - 20);
        ctx.lineTo(width/2, height/2 + 20);
        ctx.moveTo(width/2 - 20, height/2);
        ctx.lineTo(width/2 + 20, height/2);
        ctx.stroke();
        
        // Draw rectangle guide (80% of frame size)
        const rectWidth = width * 0.8;
        const rectHeight = height * 0.8;
        const x = (width - rectWidth) / 2;
        const y = (height - rectHeight) / 2;
        
        ctx.beginPath();
        ctx.rect(x, y, rectWidth, rectHeight);
        ctx.stroke();
        
        // Draw horizontal level line
        ctx.beginPath();
        ctx.moveTo(width/4, height/2);
        ctx.lineTo(3*width/4, height/2);
        ctx.stroke();
    }
    
    // Start processing frames
    function startFrameProcessing() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
        }
        
        function processFrame(timestamp) {
            if (!stream) return;
            
            // Only analyze every ANALYSIS_INTERVAL ms
            if (!processingFrame && (timestamp - lastAnalysisTime) >= ANALYSIS_INTERVAL) {
                lastAnalysisTime = timestamp;
                analyzeCurrentFrame();
            }
            
            animationFrameId = requestAnimationFrame(processFrame);
        }
        
        animationFrameId = requestAnimationFrame(processFrame);
    }
    
    // Analyze the current camera frame
    async function analyzeCurrentFrame() {
        if (processingFrame || !stream) return;
        
        // Check if video element is actually displaying (tab is active)
        if (document.hidden || !cameraView.videoWidth) {
            return;
        }
        
        processingFrame = true;
        
        try {
            // Draw the current video frame to the hidden canvas
            const ctx = cameraCanvas.getContext('2d');
            ctx.drawImage(cameraView, 0, 0, cameraCanvas.width, cameraCanvas.height);
            
            // Convert canvas to base64 image
            const imageData = cameraCanvas.toDataURL('image/jpeg', 0.8);
            
            // Send to server for analysis
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData
                })
            });
            
            if (!response.ok) {
                throw new Error('Server error: ' + response.status);
            }
            
            // Reset error count on successful response
            errorCount = 0;
            networkStatus = true;
            
            const result = await response.json();
            
            if (result.success) {
                updateMetrics(result.metrics);
            } else {
                console.error('Analysis failed:', result.error);
            }
        } catch (error) {
            console.error('Error analyzing frame:', error);
            // Only show error to user if it persists
            if (++errorCount > 3) {
                updateStatus('Connection to server lost. Please refresh the page.', 'error');
                // Check if server is still responsive
                await checkServerConnection();
            }
        } finally {
            processingFrame = false;
        }
    }
    
    // Update metrics display
    function updateMetrics(metrics) {
        if (!metrics) return;
        
        latestMetrics = metrics;
        
        // Update progress bars
        lightingProgress.style.width = `${metrics.darkness_index}%`;
        lightingProgress.textContent = `${Math.round(metrics.darkness_index)}%`;
        lightingProgress.className = 'progress-bar ' + 
            (metrics.darkness_index >= QUALITY_THRESHOLD ? 'bg-success' : 'bg-danger');
        
        angleProgress.style.width = `${metrics.angle_index}%`;
        angleProgress.textContent = `${Math.round(metrics.angle_index)}%`;
        angleProgress.className = 'progress-bar ' + 
            (metrics.angle_index >= QUALITY_THRESHOLD ? 'bg-success' : 'bg-danger');
        
        iqiProgress.style.width = `${metrics.iqi_score}%`;
        iqiProgress.textContent = `${Math.round(metrics.iqi_score)}%`;
        iqiProgress.className = 'progress-bar ' + 
            (metrics.iqi_score >= QUALITY_THRESHOLD ? 'bg-success' : 'bg-danger');
        
        // Update text values
        lightingValue.textContent = `${metrics.darkness_index.toFixed(1)}/100`;
        angleValue.textContent = `${metrics.angle_index.toFixed(1)}/100 (${metrics.angle})`;
        iqiValue.textContent = `${metrics.iqi_score.toFixed(1)}/100`;
        
        // Update status message
        if (metrics.iqi_score >= QUALITY_THRESHOLD) {
            updateStatus('Good quality! Press CAPTURE to take a photo.', 'success');
            captureBtn.disabled = false;
        } else {
            let message = 'Adjust ';
            let issues = [];
            
            if (metrics.darkness_index < QUALITY_THRESHOLD) {
                issues.push('lighting');
            }
            
            if (metrics.angle_index < QUALITY_THRESHOLD) {
                issues.push('angle');
            }
            
            message += issues.join(' and ') + ' for better quality.';
            updateStatus(message, 'warning');
        }
    }
    
    // Update status message
    function updateStatus(message, type = '') {
        statusMessage.innerHTML = `<p>${message}</p>`;
        statusMessage.className = 'status-message';
        
        if (type) {
            statusMessage.classList.add(type);
        }
    }
    
    // Capture current frame
    async function captureImage() {
        if (!stream || !latestMetrics) return;
        
        try {
            // Show flash effect
            flash.classList.add('flash-active');
            setTimeout(() => {
                flash.classList.remove('flash-active');
            }, 300);
            
            // Show loading during save
            loadingIndicator.classList.add('active');
            
            // Get current frame
            const ctx = cameraCanvas.getContext('2d');
            ctx.drawImage(cameraView, 0, 0, cameraCanvas.width, cameraCanvas.height);
            const imageData = cameraCanvas.toDataURL('image/jpeg', 0.9);
            
            // Save the image on the server
            const response = await fetch('/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData,
                    metrics: latestMetrics
                })
            });
            
            // Hide loading
            loadingIndicator.classList.remove('active');
            
            if (!response.ok) {
                throw new Error('Failed to save image');
            }
            
            const result = await response.json();
            
            if (result.success) {
                // Add image to gallery
                addCapturedImage(result.filename, latestMetrics);
                updateStatus('Image captured and saved successfully!', 'success');
            } else {
                updateStatus('Failed to save image: ' + result.error, 'error');
            }
        } catch (error) {
            // Hide loading if error
            loadingIndicator.classList.remove('active');
            
            console.error('Error capturing image:', error);
            updateStatus('Error capturing image: ' + error.message, 'error');
        }
    }
    
    // Add captured image to gallery
    function addCapturedImage(filename, metrics) {
        const imageContainer = document.createElement('div');
        imageContainer.className = 'captured-image';
        
        const img = document.createElement('img');
        img.src = `/images/${filename}`;
        img.alt = 'Captured chiller';
        
        const infoDiv = document.createElement('div');
        infoDiv.className = 'image-info';
        infoDiv.textContent = `IQI: ${metrics.iqi_score.toFixed(1)}/100`;
        
        imageContainer.appendChild(img);
        imageContainer.appendChild(infoDiv);
        
        // Add to the beginning of the container
        if (capturedImagesContainer.firstChild) {
            capturedImagesContainer.insertBefore(imageContainer, capturedImagesContainer.firstChild);
        } else {
            capturedImagesContainer.appendChild(imageContainer);
        }
    }
    
    // Event listeners
    startCameraBtn.addEventListener('click', async () => {
        if (!availableCameras.length) {
            const initialized = await initCamera();
            if (!initialized) return;
        }
        
        await startCamera();
    });
    
    captureBtn.addEventListener('click', captureImage);
    
    switchCameraBtn.addEventListener('click', switchCamera);
    
    // Initialize
    (async function initialize() {
        // Check server connection
        await checkServerConnection();
        
        // Define errorCount for tracking connection issues
        window.errorCount = 0;
        
        // Check if running on mobile
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        if (isMobile) {
            // On mobile, enable the switch camera button regardless of detected cameras
            switchCameraBtn.disabled = false;
            switchCameraBtn.textContent = "Use Back Camera"; // Default is front camera
            document.body.classList.add('mobile-device');
            
            // Initialize but don't check for cameras since we'll use facingMode
            try {
                availableCameras = await getAvailableCameras();
                console.log(`Detected ${availableCameras.length} cameras`);
            } catch (error) {
                console.error('Error detecting cameras:', error);
            }
        } else {
            await initCamera();
        }
        
        updateStatus('Click "Start Camera" to begin');
        
        // Set up periodic connection check (every 10 seconds)
        setInterval(checkServerConnection, 10000);
        
        // Listen for page visibility changes to pause/resume processing
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                // Page is hidden, no need to process frames
                if (animationFrameId) {
                    cancelAnimationFrame(animationFrameId);
                    animationFrameId = null;
                }
            } else if (stream) {
                // Page is visible again, resume processing
                startFrameProcessing();
            }
        });
        
        // Handle device orientation changes (mobile)
        if (window.matchMedia("(max-width: 768px)").matches) {
            window.addEventListener('orientationchange', async function() {
                // Wait for the orientation change to complete
                await new Promise(resolve => setTimeout(resolve, 300));
                
                if (stream) {
                    // Restart camera to adjust to new orientation
                    updateStatus('Adjusting to new orientation...', 'warning');
                    await startCamera();
                }
            });
            
            // Add class to body for mobile devices
            document.body.classList.add('mobile-device');
        }
    })();
}); 