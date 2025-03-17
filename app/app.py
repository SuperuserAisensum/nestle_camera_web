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
import tempfile
import requests
from roboflow import Roboflow

# Roboflow settings
ROBOFLOW_API_KEY = "Otg64Ra6wNOgDyjuhMYU"
ROBOFLOW_WORKSPACE = "alat-pelindung-diri"
ROBOFLOW_PROJECT = "nescafe-4base"
ROBOFLOW_VERSION = 89

# OWLv2 settings
OWLV2_API_KEY = "bjJkZXZrb2Y1cDMzMXh3OHdzbGl6OlFQOHVmS2JkZjBmQUs2bnF2OVJVdXFoNnc0ZW5kN1hH"
OWLV2_PROMPTS = ["bottle", "tetra pak", "cans", "carton drink"]

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'nestle_chiller_camera'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'captured_images')
app.config['SERVER_URL'] = os.getenv('SERVER_URL', 'http://108.137.198.68:5000/')  # Add server URL config
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Roboflow model
def initialize_models():
    """Initialize ML models with retry logic."""
    global rf, project, yolo_model
    try:
        print("Loading Roboflow project...")
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        yolo_model = project.version(ROBOFLOW_VERSION).model
        print("Roboflow model initialized")
    except Exception as e:
        print(f"Error initializing Roboflow model: {e}")
        raise

# Initialize models
try:
    initialize_models()
except Exception as e:
    print(f"Could not initialize models: {e}")
    # Continue without model, we'll log errors when detection is attempted

def is_overlap(box1, boxes2, threshold=0.3):
    """Check if box1 overlaps with any box in boxes2."""
    x1_min, y1_min, x1_max, y1_max = box1
    for b2 in boxes2:
        x2, y2, w2, h2 = b2
        x2_min = x2 - w2/2
        x2_max = x2 + w2/2
        y2_min = y2 - h2/2
        y2_max = y2 + h2/2

        dx = min(x1_max, x2_max) - max(x1_min, x2_min)
        dy = min(y1_max, y2_max) - max(y1_min, y2_min)
        if (dx >= 0) and (dy >= 0):
            area_overlap = dx * dy
            area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
            if area_box1 > 0 and (area_overlap / area_box1) > threshold:
                return True
    return False

def owl_detection_raw(temp_path):
    """Perform the OWLv2 detection API call and return raw detection results."""
    try:
        headers = {"Authorization": "Basic " + OWLV2_API_KEY}
        data = {"prompts": OWLV2_PROMPTS, "model": "owlv2"}
        with open(temp_path, "rb") as f:
            files = {"image": f}
            response = requests.post("https://api.landing.ai/v1/tools/text-to-object-detection",
                                   files=files, data=data, headers=headers)
        owlv2_result = response.json()
        return owlv2_result['data'][0] if 'data' in owlv2_result and len(owlv2_result['data']) > 0 else []
    except Exception as e:
        print(f"Error in OWLv2 detection: {e}")
        return []

def process_image(image_path):
    """
    Process an image by calling Roboflow and OWLv2 detection.
    Returns annotated image and detection data.
    """
    try:
        # Check if models were initialized
        if 'yolo_model' not in globals():
            print("Models not initialized, cannot perform detection")
            return None, None

        # Predict with Roboflow YOLO model
        try:
            rf_result = yolo_model.predict(image_path, confidence=50, overlap=80).json()
            print(f"Roboflow detection succeeded: {len(rf_result.get('predictions', []))} predictions")
        except Exception as e:
            print(f"Roboflow detection failed: {e}")
            rf_result = {"predictions": []}

        # Process Roboflow results
        nestle_class_count = {}
        nestle_boxes = []
        for pred in rf_result.get('predictions', []):
            cls = pred['class']
            nestle_class_count[cls] = nestle_class_count.get(cls, 0) + 1
            nestle_boxes.append((pred['x'], pred['y'], pred['width'], pred['height']))
        total_nestle = sum(nestle_class_count.values())

        # Detect with OWLv2
        owl_raw = owl_detection_raw(image_path)
        
        # Process OWLv2 results
        unclassified_boxes = []
        for obj in owl_raw:
            if 'bounding_box' in obj:
                score = obj.get("score", 0)
                if score < 0.3:  # Filter low confidence
                    continue
                bbox = obj['bounding_box']
                if not is_overlap(bbox, nestle_boxes):
                    unclassified_boxes.append({
                        "class": "unclassified",
                        "box": bbox,
                        "confidence": score
                    })
        total_unclassified = len(unclassified_boxes)

        # Draw bounding boxes on the image
        cv_img = cv2.imread(image_path)
        for pred in rf_result.get('predictions', []):
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_img, f"{pred['class']} {pred['confidence']:.2f}",
                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for comp in unclassified_boxes:
            x1, y1, x2, y2 = comp['box']
            cv2.rectangle(cv_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(cv_img, f"unclassified {comp['confidence']:.2f}",
                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Prepare detection data
        detection_data = {
            "roboflow_predictions": nestle_class_count,
            "total_nestle": total_nestle,
            "total_unclassified": total_unclassified,
            "unclassified_detections": unclassified_boxes
        }

        print("\nDetection Summary:")
        print("Nestlé Products:", nestle_class_count)
        print("Unclassified Products:", total_unclassified)
        
        return cv_img, detection_data
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return None, None

# New function to detect objects in base64 image before capture
def detect_objects_in_base64(image_data):
    """
    Detect objects in a base64 image without saving it.
    Returns detection data and a boolean indicating if products were detected.
    """
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name
        
        try:
            # Use the existing detection logic
            _, detection_data = process_image(temp_path)
            
            if detection_data is None:
                return {"total_nestle": 0, "total_unclassified": 0}, False
            
            # Check if any products were detected
            has_products = (detection_data["total_nestle"] > 0 or detection_data["total_unclassified"] > 0)
            
            return detection_data, has_products
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"Error detecting objects in base64 image: {e}")
        traceback.print_exc()
        return {"total_nestle": 0, "total_unclassified": 0}, False

class ChillerImageProcessor:
    def __init__(self):
        self.focal_length = 1200  # Default focal length for angle calculations

    ############################################################################
    # All processing below uses BGR format for OpenCV. We do ONE conversion
    # from RGB -> BGR after decoding the base64 image. This avoids color shifts.
    ############################################################################

    def compute_darkness_index(self, image_bgr):
        """
        Calculate darkness index from 0-100, using BGR input (OpenCV).
        """
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
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
        """
        Calculate intersection point of two lines in BGR image shape.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        h, w = image_shape

        x1, y1 = x1 / w, y1 / h
        x2, y2 = x2 / w, y2 / h
        x3, y3 = x3 / w, y3 / h
        x4, y4 = x4 / w, y4 / h

        A1, B1, C1 = y2 - y1, x1 - x2, (y2 - y1) * x1 + (x1 - x2) * y1
        A2, B2, C2 = y4 - y3, x3 - x4, (y4 - y3) * x3 + (x3 - x4) * y3
        determinant = A1 * B2 - A2 * B1

        if abs(determinant) < 1e-10:
            return None

        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant

        x, y = x * w, y * h
        return (x, y)

    def find_vanishing_point(self, image_bgr):
        """
        Find the vanishing point in a BGR image with more tolerant parameters.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 30, 150, apertureSize=3)  # Menurunkan threshold bawah

        # Menurunkan threshold dan panjang minimum garis
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, 
                               minLineLength=100, maxLineGap=15)
        h, w = image_bgr.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        vp = (cx, cy)
        filtered_lines = []
        intersections = []

        if lines is not None and len(lines) >= 2:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                # Memperlebar range sudut yang diterima
                if (10 < angle < 80 or 100 < angle < 170) and length > 100:
                    filtered_lines.append(line[0])

            if len(filtered_lines) >= 2:
                # Memperbesar margin dan threshold
                margin = w * 0.75
                center_threshold = w * 0.4
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

        return vp

    def compute_angle_from_vanishing_point(self, vp, image_shape):
        """
        Calculate angle from the vanishing point in degrees, BGR shape.
        """
        h, w = image_shape
        cx, cy = w / 2.0, h / 2.0
        dx = vp[0] - cx
        dy = vp[1] - cy
        distance = math.sqrt(dx**2 + dy**2)
        angle_rad = math.atan2(distance, self.focal_length)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def compute_angle_score(self, angle):
        """
        Calculate angle quality score 0-100 with more tolerant scoring.
        """
        if angle is None:
            return 75  # Nilai default lebih tinggi
        # Menggunakan fungsi exponential yang lebih toleran
        score = 100 * math.exp(-angle / 25)  # Mengubah dari 15 ke 25 untuk lebih toleran
        return round(max(0, min(100, score)), 2)

    def compute_iqi_score(self, image_data):
        """
        Compute the IQI for a base64-encoded image with adjusted weights.
        70% darkness, 30% angle.
        """
        try:
            # 1) Decode base64 → PIL
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            pil_img = Image.open(BytesIO(image_bytes))

            # 2) Convert PIL (RGB) → np.array (RGB)
            image_rgb = np.array(pil_img)

            # 3) Convert from RGB → BGR once for OpenCV
            if image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_rgb
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")

        # --- Perform IQI logic in BGR ---
        darkness_index = self.compute_darkness_index(image_bgr)
        vp = self.find_vanishing_point(image_bgr)

        h, w = image_bgr.shape[:2]
        default_vp = (w / 2.0, h / 2.0)

        if vp == default_vp:
            angle = None
            angle_index = self.compute_angle_score(None)
        else:
            angle = self.compute_angle_from_vanishing_point(vp, (h, w))
            angle_index = self.compute_angle_score(angle)

        # Mengubah bobot: 70% darkness, 30% angle
        if angle is None:
            iqi_score = round((darkness_index * 0.7) + (angle_index * 0.3), 2)
            angle_text = "N/A"
        else:
            iqi_score = round((darkness_index * 0.7) + (angle_index * 0.3), 2)
            angle_text = f"{angle:.2f}°"

        return {
            'darkness_index': darkness_index,
            'angle_index': angle_index,
            'angle': angle_text,
            'iqi_score': iqi_score
        }

    def save_image(self, image_data, metrics, location=None):
        """
        Save the captured image with quality metrics and location data.
        Adjusts the image to 9:16 aspect ratio for mobile displays.
        """
        try:
            # 1) Decode base64 → PIL (RGB)
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            pil_img = Image.open(BytesIO(image_bytes))

            # 2) Convert to array for OpenCV => RGB
            image_rgb = np.array(pil_img)

            # 3) Convert from RGB -> BGR if we want to draw text with OpenCV
            if image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_rgb

            # 4) Adjust to 9:16 aspect ratio
            h, w = image_bgr.shape[:2]
            target_aspect = 9 / 16  # Aspect ratio for mobile phones
            current_aspect = w / h
            
            if current_aspect > target_aspect:  # Image is too wide
                # Calculate how much to crop from the sides
                new_width = int(h * target_aspect)
                margin = (w - new_width) // 2
                image_bgr = image_bgr[:, margin:margin+new_width]
            elif current_aspect < target_aspect:  # Image is too tall
                # Calculate how much to crop from top/bottom
                new_height = int(w / target_aspect)
                margin = (h - new_height) // 2
                image_bgr = image_bgr[margin:margin+new_height, :]
                
            # After cropping, get new dimensions
            h, w = image_bgr.shape[:2]

            # 5) Add metadata text using OpenCV in BGR
            footer_height = 60  # Height of the metadata footer
            metadata_img = np.zeros((h + footer_height, w, 3), dtype=np.uint8)
            metadata_img[0:h, 0:w] = image_bgr
            cv2.rectangle(metadata_img, (0, h), (w, h + footer_height), (0, 0, 0), -1)

            # Add metrics text
            metrics_text = f"IQI: {metrics['iqi_score']:.1f}/100 | Light: {metrics['darkness_index']:.1f}/100 | Angle: {metrics['angle']}"
            cv2.putText(metadata_img, metrics_text, (10, h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add location text if available
            location_text = "Location: "
            if location and location.get('latitude') != 'N/A':
                location_text += f"Lat: {location.get('latitude'):.6f}, Long: {location.get('longitude'):.6f}"
            else:
                location_text += "Not available"
                
            cv2.putText(metadata_img, location_text, (10, h + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 6) Convert BGR -> RGB before saving to keep normal colors
            final_rgb = cv2.cvtColor(metadata_img, cv2.COLOR_BGR2RGB)

            # 7) Save as PIL
            result_image = Image.fromarray(final_rgb)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chiller_{timestamp}_IQI{metrics['iqi_score']:.0f}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
    """Render the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Endpoint: receives base64 image JSON,
    returns IQI metrics in JSON.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        metrics = image_processor.compute_iqi_score(data['image'])
        return jsonify({'success': True, 'metrics': metrics})
    except ValueError as e:
        # Decoding / image conversion error
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect_objects():
    """
    Endpoint: receives base64 image JSON,
    returns object detection results in JSON.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Detect objects in the image
        detection_data, has_products = detect_objects_in_base64(data['image'])
        
        return jsonify({
            'success': True, 
            'detection': detection_data,
            'has_products': has_products
        })
    except ValueError as e:
        # Decoding / image conversion error
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error detecting objects: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/save', methods=['POST'])
def save_image():
    """
    Endpoint: receives base64 image JSON + metrics + location,
    saves final annotated image to disk and forwards to main server.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data or 'metrics' not in data:
            return jsonify({'error': 'Missing image or metrics data'}), 400

        # Verify that image contains products - always enforce this check
        detection_data, has_products = detect_objects_in_base64(data['image'])
        
        # If no products detected, return error
        if not has_products:
            return jsonify({
                'success': False,
                'error': 'No Nestlé or competitor products detected in the image',
                'detection': detection_data,
                'has_products': False
            }), 200  # Use 200 to ensure client gets the message
        
        # Get location data if available
        location = data.get('location', {})
        
        # Save image locally first
        filename, filepath = image_processor.save_image(
            data['image'], 
            data['metrics'],
            location
        )
        
        if not filename or not filepath:
            return jsonify({'error': 'Failed to save image locally'}), 500

        print(f"Image saved locally: {filepath}")
        
        # Process the image with object detection
        labeled_image, detection_data = process_image(filepath)
        
        if labeled_image is not None and detection_data is not None:
            # Save the labeled image
            labeled_filename = "labeled_" + filename
            labeled_filepath = os.path.join(app.config['UPLOAD_FOLDER'], labeled_filename)
            cv2.imwrite(labeled_filepath, labeled_image)
            print(f"Saved labeled image to: {labeled_filepath}")
            
            # Use the labeled image and detection data for server transmission
            filepath = labeled_filepath
            filename = labeled_filename
        else:
            # If detection failed, use the original image and create empty detection data
            print("Object detection failed, using original image")
            detection_data = {
                "roboflow_predictions": {},
                "total_nestle": 0,
                "total_unclassified": 0,
                "unclassified_detections": []
            }

        # Create form data for server request
        form_data = {
            'device_id': 'camera_web_app',
            'timestamp': datetime.now().isoformat(),
            'latitude': str(location.get('latitude', 'N/A')),
            'longitude': str(location.get('longitude', 'N/A')),
            'iqi_score': str(data['metrics']['iqi_score']),  # Mengirim IQI score yang sudah dihitung
            'roboflow_outputs': json.dumps(detection_data)
        }
        
        # Send to main server
        try:
            # Gunakan nama file yang sama dengan yang disimpan lokal
            files = {'image0': (filename, open(filepath, 'rb'), 'image/jpeg')}
            server_url = f"{app.config['SERVER_URL']}/receive_data"
            
            response = requests.post(server_url,
                                  data=form_data,
                                  files=files,
                                  timeout=30)
            
            if response.ok:
                server_result = response.json()
                # Return both local filename and server response
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'detection': detection_data,
                    'server_response': server_result
                })
            else:
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'detection': detection_data,
                    'server_error': f'Server error: {response.status_code}'
                })
                
        except requests.exceptions.RequestException as e:
            return jsonify({
                'success': True,
                'filename': filename,
                'detection': detection_data,
                'server_error': f'Connection error: {str(e)}'
            })
            
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/static/captured_images/<path:filename>')
def serve_image(filename):
    """Serve captured images from the upload folder."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Use 'adhoc' SSL context for development
    app.run(debug=True, host='0.0.0.0', port=4000, ssl_context='adhoc')
    # app.run(debug=True, host='0.0.0.0', port=4000)
