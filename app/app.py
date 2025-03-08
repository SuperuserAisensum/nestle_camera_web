import os
import cv2
import numpy as np
import math
import time
import base64
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from io import BytesIO
import json
from PIL import Image
import traceback

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'nestle_chiller_camera'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'captured_images')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class ChillerImageProcessor:
    def __init__(self):
        self.focal_length = 1200  # Default focal length for angle calculations
        
    def compute_darkness_index(self, image):
        """Calculate darkness index from 0-100"""
        # First determine if the image is BGR (from OpenCV) or RGB (from PIL)
        if isinstance(image, np.ndarray) and image.shape[2] == 3:
            # Image is likely BGR (OpenCV format)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # Already grayscale or other format
            gray_image = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        avg_intensity = np.mean(gray_image)
        darkness_index = (avg_intensity / 255) * 100
        
        if darkness_index <= 35:
            score = (darkness_index / 35) * 100
        elif 35 < darkness_index <= 55:
            score = ((darkness_index - 35) / 20) * 100
            score = 50 + (score / 2)
        elif 55 < darkness_index <= 75:
            score = ((75 - darkness_index) / 20) * 100
            score = 50 + (score / 2)
        else:
            score = ((100 - darkness_index) / 25) * 100
            
        return round(max(0, min(100, score)), 2)
    
    def compute_intersection(self, line1, line2, image_shape):
        """Calculate intersection point of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        h, w = image_shape

        # Normalize coordinates to [0, 1] range
        x1, y1 = x1 / w, y1 / h
        x2, y2 = x2 / w, y2 / h
        x3, y3 = x3 / w, y3 / h
        x4, y4 = x4 / w, y4 / h

        A1, B1, C1 = y2 - y1, x1 - x2, (y2 - y1) * x1 + (x1 - x2) * y1
        A2, B2, C2 = y4 - y3, x3 - x4, (y4 - y3) * x3 + (x3 - x4) * y3
        determinant = A1 * B2 - A2 * B1

        # Check for near-parallel lines or overflow
        if abs(determinant) < 1e-10:  # Threshold for near-zero determinant
            return None

        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant

        # Denormalize back to pixel coordinates
        x, y = x * w, y * h
        return (x, y)
    
    def find_vanishing_point(self, image):
        """Find the vanishing point to determine the angle of shot"""
        # Convert to OpenCV format if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            
        # Convert RGB to BGR for OpenCV
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, 
                              minLineLength=150, maxLineGap=10)
                              
        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        vp = (cx, cy)
        filtered_lines = []
        intersections = []

        if lines is not None and len(lines) >= 2:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                if (15 < angle < 75 or 105 < angle < 165) and length > 200:
                    filtered_lines.append(line[0])

            if len(filtered_lines) >= 2:
                margin = w / 2
                center_threshold = w / 3
                for i in range(len(filtered_lines)):
                    for j in range(i + 1, len(filtered_lines)):
                        pt = self.compute_intersection(filtered_lines[i], filtered_lines[j], (h, w))
                        if pt and -margin <= pt[0] <= w + margin and -margin <= pt[1] <= h + margin:
                            distance_to_center = math.sqrt((pt[0] - cx)**2 + (pt[1] - cy)**2)
                            if distance_to_center < center_threshold:
                                intersections.append(pt)

                if intersections:
                    intersections = np.array(intersections)
                    if len(intersections) > 3:
                        try:
                            from sklearn.cluster import KMeans
                            kmeans = KMeans(n_clusters=min(3, len(intersections)), n_init=10).fit(intersections)
                            centers = kmeans.cluster_centers_
                            distances = [math.sqrt((x - cx)**2 + (y - cy)**2) for x, y in centers]
                            best_cluster_idx = np.argmin(distances)
                            vp = tuple(centers[best_cluster_idx])
                        except Exception:
                            weights = [1 / (math.sqrt((x - cx)**2 + (y - cy)**2) + 1) for x, y in intersections]
                            vp = tuple(np.average(intersections, axis=0, weights=weights))
                    else:
                        vp = tuple(np.median(intersections, axis=0))
                        
        return vp, filtered_lines
    
    def compute_angle_from_vanishing_point(self, vp, image_shape):
        """Calculate the angle from the vanishing point"""
        h, w = image_shape
        cx, cy = w / 2.0, h / 2.0
        dx = vp[0] - cx
        dy = vp[1] - cy
        distance = math.sqrt(dx**2 + dy**2)
        angle_rad = math.atan2(distance, self.focal_length)
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    
    def compute_angle_score(self, angle):
        """Calculate score for the angle"""
        if angle is None:
            return 50
        score = 100 * math.exp(-angle / 15)
        return round(max(0, min(100, score)), 2)
    
    def compute_iqi_score(self, image):
        """Compute the Image Quality Index score"""
        # Convert base64 to image if needed
        if isinstance(image, str) and image.startswith('data:image'):
            # Strip the data URL prefix
            image_data = image.split(',')[1]
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            # Convert to PIL Image
            image = Image.open(BytesIO(image_bytes))
            # Convert to numpy array
            image = np.array(image)
            
        # Process the image
        darkness_index = self.compute_darkness_index(image)
        
        vp, filtered_lines = self.find_vanishing_point(image)
        
        # If no vanishing point detected
        if vp == (image.shape[1] / 2.0, image.shape[0] / 2.0) and not filtered_lines:
            angle = None
            angle_index = 50
        else:
            angle = self.compute_angle_from_vanishing_point(vp, image.shape[:2])
            angle_index = self.compute_angle_score(angle)

        if angle is None:
            iqi_score = round((darkness_index * 0.7) + (angle_index * 0.3), 2)
            angle_text = "N/A"
        else:
            iqi_score = round((darkness_index * 0.5) + (angle_index * 0.5), 2)
            angle_text = f"{angle:.2f}°"
            
        return {
            'darkness_index': darkness_index,
            'angle_index': angle_index,
            'angle': angle_text,
            'iqi_score': iqi_score
        }
        
    def save_image(self, image_data, metrics):
        """Save the captured image with quality metrics"""
        try:
            # Decode base64 image
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            
            # Open as PIL Image for processing
            image = Image.open(BytesIO(image_bytes))
            
            # Create a new filename with timestamp and quality info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chiller_{timestamp}_IQI{metrics['iqi_score']:.0f}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Convert to array for OpenCV processing
            image_array = np.array(image)
            
            # Add metadata text at the bottom
            h, w = image_array.shape[:2]
            cv2.rectangle(image_array, (0, h-40), (w, h), (0, 0, 0), -1)
            text = f"IQI: {metrics['iqi_score']:.1f}/100 | Light: {metrics['darkness_index']:.1f}/100 | Angle: {metrics['angle_index']:.1f}/100 ({metrics['angle']})"
            cv2.putText(image_array, text, (10, h-15), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Convert back to RGB for PIL
            if image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                
            # Save the image with metadata
            result_image = Image.fromarray(image_array)
            result_image.save(filepath)
            
            return filename, filepath
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            traceback.print_exc()
            return None, None

# Initialize the image processor
image_processor = ChillerImageProcessor()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Process and analyze an uploaded image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        # Analyze the image for quality metrics
        metrics = image_processor.compute_iqi_score(data['image'])
        
        # Return the analysis results
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/save', methods=['POST'])
def save_image():
    """Save the captured image with quality metrics"""
    try:
        data = request.get_json()
        if not data or 'image' not in data or 'metrics' not in data:
            return jsonify({'error': 'Missing image or metrics data'}), 400
            
        # Save the image
        filename, filepath = image_processor.save_image(data['image'], data['metrics'])
        
        if filename:
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': filepath
            })
        else:
            return jsonify({'error': 'Failed to save image'}), 500
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve captured images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 