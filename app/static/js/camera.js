document.addEventListener('DOMContentLoaded', () => {
    const cameraView = document.getElementById('camera-view');
    const cameraCanvas = document.getElementById('camera-canvas');
    const guidesCanvas = document.getElementById('guides-canvas');
    const cameraFeedback = document.getElementById('camera-feedback');
    const startCameraBtn = document.getElementById('start-camera');
    const captureBtn = document.getElementById('capture-btn');
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
    let processingFrame = false;
    let animationFrameId = null;
    let latestMetrics = null;
    let torchEnabled = false;
    let torchCapable = false;
    let torchTrack = null;
    let autoTorchEnabled = true; // Default auto-torch to on
    let lightingThreshold = 50; // Threshold below which torch turns on
    
    // Geolocation variables
    let currentPosition = null;
    let locationWatchId = null;
    
    // Smoothing parameters for angle (point 2)
    let smoothedAngle = null;
    const smoothingFactor = 1.0; // Higher values = more smoothing

    // Quality threshold for acceptable IQI - significantly lowered for easy success
    const QUALITY_THRESHOLD = 60;
    const ANALYSIS_INTERVAL = 500; // in milliseconds
    let lastAnalysisTime = 0;
    let errorCount = 0;
    let networkStatus = true;

    // Since only the rear camera is to be used, remove switch camera functionality.
    // Hide the switch camera button if it exists.
    const switchCameraBtn = document.getElementById('switch-camera');
    if (switchCameraBtn) {
        switchCameraBtn.style.display = 'none';
    }
    
    // Function to check server connection (kept as before)
    async function checkServerConnection() {
        try {
            const response = await fetch('/', { 
                method: 'HEAD',
                cache: 'no-store'
            });
            if (response.ok && !networkStatus) {
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
    
    // Start camera using rear camera only (point 4)
    async function startCamera() {
        try {
            if (stream) {
                stopCamera();
            }
            
            loadingIndicator.classList.add('active');
            
            // Start geolocation tracking
            startLocationTracking();
            
            // Force use of the rear camera by using facingMode "environment"
            const constraints = {
                video: {
                    facingMode: "environment",
                    width: { ideal: 1080 },
                    height: { ideal: 1920 },  // 9:16 aspect ratio (portrait orientation)
                    aspectRatio: { ideal: 9/16 }
                }
            };
            
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            cameraView.srcObject = stream;
            
            // Check if torch/flash is available
            const tracks = stream.getVideoTracks();
            if (tracks.length > 0) {
                torchTrack = tracks[0];
                const capabilities = torchTrack.getCapabilities();
                torchCapable = capabilities.torch || false;
                
                if (torchCapable) {
                    console.log("Torch capability detected");
                    // Initialize torch to off
                    await toggleTorch(false);
                } else {
                    console.log("Torch capability not available on this device");
                }
            }
            
            cameraView.onloadedmetadata = () => {
                loadingIndicator.classList.remove('active');
                cameraCanvas.width = cameraView.videoWidth;
                cameraCanvas.height = cameraView.videoHeight;
                guidesCanvas.width = cameraView.videoWidth;
                guidesCanvas.height = cameraView.videoHeight;
                
                // Initial drawing of guides
                drawGuides(null, false);
                startFrameProcessing();
            };
            
            captureBtn.disabled = false;
            startCameraBtn.textContent = 'Restart Camera';
            updateStatus('Camera started. Position the chiller within the guides.');
            return true;
        } catch (error) {
            loadingIndicator.classList.remove('active');
            console.error('Error starting camera:', error);
            updateStatus('Failed to start camera: ' + error.message, 'error');
            return false;
        }
    }
    
    function stopCamera() {
        if (stream) {
            // Turn off torch if it's on
            if (torchCapable && torchEnabled) {
                toggleTorch(false);
            }
            
            stream.getTracks().forEach(track => track.stop());
            stream = null;
            torchTrack = null;
            torchCapable = false;
            torchEnabled = false;
        }
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        // Stop geolocation tracking
        stopLocationTracking();
        
        cameraView.srcObject = null;
        captureBtn.disabled = true;
    }
    
    // Start tracking device geolocation
    function startLocationTracking() {
        if (!navigator.geolocation) {
            console.warn("Geolocation is not supported by this browser");
            return;
        }
        
        // Request permission first
        navigator.permissions.query({ name: 'geolocation' }).then(function(result) {
            if (result.state === 'granted' || result.state === 'prompt') {
                // Get initial position with timeout
                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        currentPosition = {
                            latitude: position.coords.latitude,
                            longitude: position.coords.longitude,
                        };
                        console.log("Initial location:", currentPosition);
                    },
                    (error) => {
                        console.warn("Error getting location:", error.message);
                        currentPosition = null;
                    },
                    { 
                        enableHighAccuracy: true,
                        timeout: 10000,
                        maximumAge: 0,
                    }
                );
                
                // Watch position updates
                locationWatchId = navigator.geolocation.watchPosition(
                    (position) => {
                        currentPosition = {
                            latitude: position.coords.latitude,
                            longitude: position.coords.longitude,
                        };
                        console.log("Location updated:", currentPosition);
                    },
                    (error) => {
                        console.warn("Error watching location:", error.message);
                    },
                    { 
                        enableHighAccuracy: true,
                        timeout: 10000,
                        maximumAge: 10000,
                    }
                );
            } else {
                console.warn("Geolocation permission denied");
                currentPosition = null;
            }
        });
    }
    
    // Stop tracking geolocation
    function stopLocationTracking() {
        if (locationWatchId !== null) {
            navigator.geolocation.clearWatch(locationWatchId);
            locationWatchId = null;
        }
        currentPosition = null;
    }
    
    // Draw dynamic guides (point 3): the box color changes based on IQI quality.
    function drawGuides(metrics, allMetricsGood = false) {
        const ctx = guidesCanvas.getContext('2d');
        const width = guidesCanvas.width;
        const height = guidesCanvas.height;
        ctx.clearRect(0, 0, width, height);
        
        // Use allMetricsGood parameter to determine guide color
        let guideColor = '#FF0000';
        if (allMetricsGood) {
            guideColor = '#00FF00';
        }
        
        ctx.strokeStyle = guideColor;
        ctx.lineWidth = 3;
        
        // Draw center crosshair
        ctx.beginPath();
        ctx.moveTo(width/2, height/2 - 20);
        ctx.lineTo(width/2, height/2 + 20);
        ctx.moveTo(width/2 - 20, height/2);
        ctx.lineTo(width/2 + 20, height/2);
        ctx.stroke();
        
        // Draw rectangle guide (80% of canvas)
        const rectWidth = width * 0.8;
        const rectHeight = height * 0.8;
        const x = (width - rectWidth) / 2;
        const y = (height - rectHeight) / 2;
        ctx.strokeRect(x, y, rectWidth, rectHeight);
        
        // Draw horizontal level line
        ctx.beginPath();
        ctx.moveTo(width/4, height/2);
        ctx.lineTo(3 * width/4, height/2);
        ctx.stroke();
    }
    
    // Process frames at regular intervals
    function startFrameProcessing() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
        }
        
        function processFrame(timestamp) {
            if (!stream) return;
            if (!processingFrame && (timestamp - lastAnalysisTime) >= ANALYSIS_INTERVAL) {
                lastAnalysisTime = timestamp;
                analyzeCurrentFrame();
            }
            animationFrameId = requestAnimationFrame(processFrame);
        }
        
        animationFrameId = requestAnimationFrame(processFrame);
    }
    
    // Analyze the current camera frame by sending it to the server for IQI computation.
    async function analyzeCurrentFrame() {
        if (processingFrame || !stream) return;
        
        if (document.hidden || !cameraView.videoWidth) return;
        
        processingFrame = true;
        
        try {
            const ctx = cameraCanvas.getContext('2d');
            ctx.drawImage(cameraView, 0, 0, cameraCanvas.width, cameraCanvas.height);
            const imageData = cameraCanvas.toDataURL('image/jpeg', 0.8);
            
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageData })
            });
            
            if (!response.ok) {
                throw new Error('Server error: ' + response.status);
            }
            
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
            if (++errorCount > 3) {
                updateStatus('Connection to server lost. Please refresh the page.', 'error');
                await checkServerConnection();
            }
        } finally {
            processingFrame = false;
        }
    }
    
    // Update metrics, apply smoothing to angle, update progress bars, and update dynamic guides.
    function updateMetrics(metrics) {
        // Process angle: if numeric, apply exponential smoothing.
        let currentAngle = parseFloat(metrics.angle);
        if (!isNaN(currentAngle)) {
            if (smoothedAngle === null) {
                smoothedAngle = currentAngle;
            } else {
                smoothedAngle = smoothingFactor * smoothedAngle + (1 - smoothingFactor) * currentAngle;
            }
            metrics.angle = smoothedAngle.toFixed(2) + '°';
        }
    
        // Update global latestMetrics so the capture function uses the latest values.
        latestMetrics = metrics;
        
        // Check lighting and manage auto-torch
        manageAutoTorch(metrics.darkness_index);
    
        // Update progress bars and text values for lighting, angle, and overall IQI.
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
    
        lightingValue.textContent = `${metrics.darkness_index.toFixed(1)}/100`;
        angleValue.textContent = `${metrics.angle_index.toFixed(1)}/100 (${metrics.angle})`;
        iqiValue.textContent = `${metrics.iqi_score.toFixed(1)}/100`;
    
        // Check if ALL metrics are above threshold
        const allMetricsGood = 
            metrics.darkness_index >= QUALITY_THRESHOLD && 
            metrics.angle_index >= QUALITY_THRESHOLD && 
            metrics.iqi_score >= QUALITY_THRESHOLD;
    
        // Enable capture button only if ALL metrics meet the threshold
        if (allMetricsGood) {
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
            if (issues.length === 0) {
                message = 'Waiting for better overall quality...';
            } else {
                message += issues.join(' and ') + ' for better quality.';
            }
            updateStatus(message, 'warning');
            captureBtn.disabled = true;
        }
        
        // Update drawGuides to use the same condition
        drawGuides(metrics, allMetricsGood);
    }    
    
    function updateStatus(message, type = '') {
        statusMessage.innerHTML = `<p class="mb-0"><strong>Status:</strong> ${message}</p>`;
        
        // Reset classes and add bootstrap alert classes
        statusMessage.className = 'status-message alert';
        
        // Map our status types to Bootstrap alert types
        if (type === 'success') {
            statusMessage.classList.add('alert-success');
        } else if (type === 'error') {
            statusMessage.classList.add('alert-danger');
        } else if (type === 'warning') {
            statusMessage.classList.add('alert-warning');
        } else {
            statusMessage.classList.add('alert-info');
        }
    }
    
    // Function to add a captured image to the gallery
    function addCapturedImage(filename, metrics, detection) {
        // Create a container for the captured image
        const imageContainer = document.createElement('div');
        imageContainer.className = 'captured-image';
        
        // Create the image element
        const img = document.createElement('img');
        img.src = `/static/captured_images/${filename}`;
        img.alt = 'Captured Image';
        img.className = 'img-fluid';
        
        // Create info container
        const infoContainer = document.createElement('div');
        infoContainer.className = 'image-info';
        
        // Add IQI score
        const iqiInfo = document.createElement('div');
        iqiInfo.className = 'iqi-info';
        iqiInfo.innerHTML = `<strong>IQI:</strong> ${metrics.iqi_score.toFixed(1)}/100`;
        
        // Add detection info if available
        let detectionInfo = '';
        if (detection && (detection.total_nestle > 0 || detection.total_unclassified > 0)) {
            detectionInfo = `
                <div class="detection-info">
                    <div><strong>Nestlé Products:</strong> ${detection.total_nestle}</div>
                    <div><strong>Unclassified:</strong> ${detection.total_unclassified}</div>
                </div>
            `;
            
            // Add product breakdown if available
            if (detection.roboflow_predictions && Object.keys(detection.roboflow_predictions).length > 0) {
                let productList = '<div class="product-list"><strong>Products:</strong><ul>';
                for (const [product, count] of Object.entries(detection.roboflow_predictions)) {
                    productList += `<li>${product}: ${count}</li>`;
                }
                productList += '</ul></div>';
                detectionInfo += productList;
            }
        } else {
            detectionInfo = '<div class="detection-info">No products detected</div>';
        }
        
        // Add location info
        const locationInfo = document.createElement('div');
        locationInfo.className = 'location-info';
        
        if (currentPosition) {
            locationInfo.innerHTML = `
                <strong>Location:</strong> 
                <div>Lat: ${currentPosition.latitude.toFixed(6)}</div>
                <div>Long: ${currentPosition.longitude.toFixed(6)}</div>
            `;
        } else {
            locationInfo.innerHTML = '<strong>Location:</strong> Not available';
        }
        
        // Assemble the info container
        infoContainer.appendChild(iqiInfo);
        infoContainer.innerHTML += detectionInfo;
        infoContainer.appendChild(locationInfo);
        
        // Add elements to the image container
        imageContainer.appendChild(img);
        imageContainer.appendChild(infoContainer);
        
        // Add the image container to the gallery
        if (capturedImagesContainer) {
            // Insert at the beginning to show newest first
            capturedImagesContainer.insertBefore(imageContainer, capturedImagesContainer.firstChild);
        }
    }
    
    // Function to capture an image
    async function captureImage() {
        if (!stream) {
            updateStatus('Camera not started', 'error');
            return;
        }
        
        try {
            // Disable the capture button during processing
            captureBtn.disabled = true;
            updateStatus('Processing image...', 'info');
            
            // Show flash effect
            flash.style.display = 'block';
            setTimeout(() => {
                flash.style.display = 'none';
            }, 500);
            
            // Draw the current frame to the canvas
            const context = cameraCanvas.getContext('2d');
            context.drawImage(cameraView, 0, 0, cameraCanvas.width, cameraCanvas.height);
            
            // Get the image data as base64
            const imageData = cameraCanvas.toDataURL('image/jpeg', 0.9);
            
            // Check if we have location data
            if (!currentPosition) {
                console.warn('Location data not available, attempting to get it now');
                try {
                    await new Promise((resolve, reject) => {
                        navigator.geolocation.getCurrentPosition(
                            (position) => {
                                currentPosition = {
                                    latitude: position.coords.latitude,
                                    longitude: position.coords.longitude
                                };
                                resolve();
                            },
                            (error) => {
                                console.warn('Could not get location:', error.message);
                                resolve(); // Resolve anyway to continue
                            },
                            { timeout: 5000, maximumAge: 0 }
                        );
                    });
                } catch (e) {
                    console.warn('Error getting location:', e);
                }
            }
            
            // Send the image to the server
            const response = await fetch('/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData,
                    metrics: latestMetrics,
                    location: currentPosition
                })
            });
            
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                updateStatus('Image captured and saved successfully!', 'success');
                // Add the image to the gallery with detection data
                addCapturedImage(result.filename, latestMetrics, result.detection);
            } else {
                updateStatus(`Error: ${result.error || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            console.error('Error capturing image:', error);
            updateStatus(`Error: ${error.message}`, 'error');
        } finally {
            // Re-enable the capture button
            captureBtn.disabled = false;
        }
    }
    
    // Event listeners for start and capture
    startCameraBtn.addEventListener('click', async () => {
        await startCamera();
    });
    
    captureBtn.addEventListener('click', captureImage);
    
    // Periodically check server connection
    setInterval(checkServerConnection, 10000);
    
    // Pause frame processing when tab is inactive
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
        } else if (stream) {
            startFrameProcessing();
        }
    });
    
    // Initialize camera on page load
    (async function initialize() {
        await checkServerConnection();
        updateStatus('Click "Start Camera" to begin');
    })();

    // Function to toggle torch/flash
    async function toggleTorch(forceState = null) {
        if (!torchCapable || !torchTrack) return false;
        
        try {
            // If forceState is provided, use that, otherwise toggle the current state
            const newState = (forceState !== null) ? forceState : !torchEnabled;
            
            // Apply the new torch state
            await torchTrack.applyConstraints({
                advanced: [{ torch: newState }]
            });
            
            torchEnabled = newState;
            console.log(`Torch ${torchEnabled ? 'enabled' : 'disabled'}`);
            
            // Update status with torch info
            if (torchEnabled) {
                updateStatus('Auto-flash enabled due to low lighting', 'info');
            }
            
            return true;
        } catch (error) {
            console.error('Error toggling torch:', error);
            return false;
        }
    }
    
    // Function to check and manage auto torch based on lighting
    function manageAutoTorch(lightingScore) {
        if (!autoTorchEnabled || !torchCapable) return;
        
        if (lightingScore < lightingThreshold && !torchEnabled) {
            toggleTorch(true);
        } else if (lightingScore >= lightingThreshold + 10 && torchEnabled) {
            // Add hysteresis to prevent toggling too frequently
            toggleTorch(false);
        }
    }
});
