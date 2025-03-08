import cv2
import numpy as np
import math
import os
import time
from datetime import datetime

class ChillerCameraApp:
    def __init__(self):
        self.camera = None
        self.window_name = "Chiller Camera IQI"
        self.output_dir = "captured_images"
        self.focal_length = 1200  # Default focal length for angle calculations
        self.frame_width = 1280
        self.frame_height = 720
        self.guide_color = (0, 255, 0)  # Green
        self.guide_thickness = 2
        self.feedback_position = (20, 40)
        self.text_color = (255, 255, 255)
        self.text_bg_color = (0, 0, 0)
        self.text_size = 0.7
        self.text_thickness = 2
        
        # Thresholds for good quality
        self.min_iqi_score = 70
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def start_camera(self, camera_id=0):
        """Initialize and start the camera feed"""
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            print("Error: Could not open camera.")
            return False
            
        # Set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        return True
    
    def compute_darkness_index(self, image):
        """Calculate darkness index from 0-100"""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

        # Add visualization of filtered lines
        vis_image = image.copy()
        for line in filtered_lines:
            x1, y1, x2, y2 = line
            cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
        # Mark the vanishing point
        cv2.circle(vis_image, (int(vp[0]), int(vp[1])), 15, (255, 0, 0), -1)
            
        return vp, vis_image, filtered_lines
    
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
    
    def compute_iqi_score(self, frame):
        """Compute the Image Quality Index score"""
        darkness_index = self.compute_darkness_index(frame)
        
        vp, vis_frame, filtered_lines = self.find_vanishing_point(frame)
        
        # If no vanishing point detected
        if vp == (frame.shape[1] / 2.0, frame.shape[0] / 2.0) and not filtered_lines:
            angle = None
            angle_index = 50
        else:
            angle = self.compute_angle_from_vanishing_point(vp, frame.shape[:2])
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
        }, vis_frame
    
    def draw_chiller_guide(self, frame):
        """Draw guide lines for chiller positioning"""
        h, w = frame.shape[:2]
        
        # Draw center crosshair
        cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), self.guide_color, self.guide_thickness)
        cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), self.guide_color, self.guide_thickness)
        
        # Draw rectangle guide (80% of frame size)
        rect_w = int(w * 0.8)
        rect_h = int(h * 0.8)
        x1 = (w - rect_w) // 2
        y1 = (h - rect_h) // 2
        cv2.rectangle(frame, (x1, y1), (x1+rect_w, y1+rect_h), self.guide_color, self.guide_thickness)
        
        # Draw horizontal level line
        cv2.line(frame, (w//4, h//2), (3*w//4, h//2), self.guide_color, self.guide_thickness)
        
        return frame
    
    def display_feedback(self, frame, metrics):
        """Display quality metrics on frame"""
        h, w = frame.shape[:2]
        y_pos = self.feedback_position[1]
        
        # Create semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 150), self.text_bg_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Display metrics
        dark_color = (0, 255, 0) if metrics['darkness_index'] >= 70 else (0, 0, 255)
        angle_color = (0, 255, 0) if metrics['angle_index'] >= 70 else (0, 0, 255)
        iqi_color = (0, 255, 0) if metrics['iqi_score'] >= 70 else (0, 0, 255)
        
        cv2.putText(frame, f"Lighting: {metrics['darkness_index']:.1f}/100", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, dark_color, self.text_thickness)
        y_pos += 30
        
        cv2.putText(frame, f"Angle: {metrics['angle_index']:.1f}/100 ({metrics['angle']})", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, angle_color, self.text_thickness)
        y_pos += 30
        
        cv2.putText(frame, f"Overall: {metrics['iqi_score']:.1f}/100", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, iqi_color, self.text_thickness)
        y_pos += 30
        
        if metrics['iqi_score'] >= self.min_iqi_score:
            cv2.putText(frame, "PRESS SPACE TO CAPTURE", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, (0, 255, 0), self.text_thickness)
        else:
            cv2.putText(frame, "ADJUST POSITION/LIGHTING", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, (0, 0, 255), self.text_thickness)
            
        return frame
    
    def save_image(self, frame, metrics):
        """Save the captured image with quality metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chiller_{timestamp}_IQI{metrics['iqi_score']:.0f}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        # Add quality info to the image
        info_frame = frame.copy()
        h, w = info_frame.shape[:2]
        
        # Add metadata text at the bottom
        cv2.rectangle(info_frame, (0, h-40), (w, h), (0, 0, 0), -1)
        text = f"IQI: {metrics['iqi_score']:.1f}/100 | Light: {metrics['darkness_index']:.1f}/100 | Angle: {metrics['angle_index']:.1f}/100 ({metrics['angle']})"
        cv2.putText(info_frame, text, (10, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(filepath, info_frame)
        print(f"Image saved: {filepath}")
        return filepath
    
    def run(self):
        """Main application loop"""
        if not self.start_camera():
            return
            
        print("Camera started. Press 'q' to quit, SPACE to capture.")
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                    
                # Process frame
                metrics, vis_frame = self.compute_iqi_score(frame)
                
                # Draw guides and feedback
                guide_frame = self.draw_chiller_guide(vis_frame)
                display_frame = self.display_feedback(guide_frame, metrics)
                
                # Show the frame
                cv2.imshow(self.window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Spacebar to capture
                    self.save_image(frame, metrics)
                    # Flash effect
                    white_flash = np.ones(frame.shape, dtype=np.uint8) * 255
                    cv2.imshow(self.window_name, white_flash)
                    cv2.waitKey(100)
                    
        finally:
            # Clean up
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            print("Camera closed.")

if __name__ == "__main__":
    app = ChillerCameraApp()
    app.run() 