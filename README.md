# Nestle Chiller Camera IQI Web Application

This web application provides a real-time camera interface for taking high-quality images of chillers with quality assessment for:
- Lighting conditions
- Angle of shot
- Chiller positioning

## Features

- Live camera feed with positioning guides
- Real-time quality feedback (lighting, angle, overall score)
- Automatic quality scoring
- Image saving with embedded quality information
- Visual feedback for proper positioning
- Support for multiple cameras/switching cameras
- Gallery of captured images

## How to Use

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   pip install pyngrok
   ```

2. Run the web application:
   ```
   python app/app.py
   ```

3. Open the application in your browser:
   ```
   http://localhost:5000
   ```

4. Interface controls:
   - Click "Start Camera" to initialize your webcam
   - Position the chiller within the green rectangle guide
   - The application will analyze the frame quality in real-time
   - When quality metrics are good (≥70%), click "Capture" to take a photo
   - Use "Switch Camera" if you have multiple cameras connected

5. Quality metrics explanation:
   - Lighting Quality: Measures if the image has proper lighting
   - Angle Quality: Measures if the chiller is photographed from a proper angle
   - Overall IQI Score: Combined quality score weighted by lighting and angle

6. Captured images are saved in the `captured_images` directory with quality scores in the filename.

## Technical Requirements

- Python 3.7+
- Webcam or external camera
- Modern web browser with camera access permissions
- Packages listed in requirements.txt

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Edge
- Safari (limited support)

## Note for Mobile Devices

The application can be accessed on mobile devices when running on a network-accessible server. Use the device's IP address instead of localhost, e.g.:
```
http://192.168.1.100:5000
```

## Quality Metrics

- **Lighting Index (0-100)**: Measures if the image has proper lighting
- **Angle Index (0-100)**: Measures if the chiller is photographed from a proper angle
- **Overall IQI Score (0-100)**: Combined quality score weighted by lighting and angle

A score of 70 or higher is considered sufficient quality for each metric. 
