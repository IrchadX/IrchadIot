#!/usr/bin/env python3

import time
import queue
import threading
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import pyttsx3

@dataclass
class DetectedObstacle:
    type: str  # "vision" or "ultrasonic"
    object_class: str  # class name or "obstacle"
    confidence: float
    distance: float  # meters
    angle: float  # degrees from center (-90 to +90)
    position: Tuple[float, float]  # relative (x, y) position
    timestamp: float
    size: Tuple[float, float] = (0.0, 0.0)  # width, height in meters
    
@dataclass
class ObstacleDetectionConfig:
    # Camera settings
    camera_resolution: Tuple[int, int] = (320, 320)
    yolo_model_path: str = "yoloe-11s-det.pt"
    vision_conf_threshold: float = 0.3
    camera_fov: float = 62.2  # degrees, typical for Pi camera
    
    # Ultrasonic settings
    ultrasonic_trigger_pin: int = 23
    ultrasonic_echo_pin: int = 24
    ultrasonic_max_distance: float = 4.0  # meters
    ultrasonic_min_distance: float = 0.02  # meters
    
    # Detection settings
    detection_rate: float = 10.0  # Hz
    obstacle_memory_time: float = 5.0  # seconds
    min_obstacle_size: float = 0.1  # meters
    max_obstacle_distance: float = 5.0  # meters
    
    # Safety thresholds
    critical_distance: float = 0.5  # meters
    warning_distance: float = 1.5  # meters
    path_blocking_threshold: float = 0.8  # meters from path

class VisionObstacleDetector:
    def __init__(self, config: ObstacleDetectionConfig, shared_camera=None):
        self.config = config
        self.shared_camera = shared_camera  # Use shared camera if provided
        self.camera = None
        self.model = None
        self.is_running = False
        self.owns_camera = shared_camera is None  # Track if we own the camera
        
        # YOLO classes for obstacle detection
        self.obstacle_classes = [
            "stairs", "wall", "door", "furniture", "person", "table", "chair", 
            "backpack", "AC", "curtain", "tree"
        ]
        
        # Load font for visualization
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except:
            self.font = ImageFont.load_default()
    
    def initialize(self) -> bool:
        """Initialize camera and YOLO model"""
        try:
            # Use shared camera if available, otherwise create our own
            if self.shared_camera is not None:
                self.camera = self.shared_camera
                print("VisionObstacleDetector: Using shared camera instance")
            else:
                self.camera = Picamera2()
                camera_config = self.camera.create_still_configuration(
                    main={"size": self.config.camera_resolution}
                )
                self.camera.configure(camera_config)
                self.camera.start()
                print("VisionObstacleDetector: Initialized new camera instance")
            
            # Initialize YOLO model
            self.model = YOLO(self.config.yolo_model_path)
            
            print("Vision obstacle detector initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize vision detector: {e}")
            return False
    def detect_obstacles(self, current_heading: float = 0.0) -> List[DetectedObstacle]:
        """Detect obstacles using camera and YOLO"""
        if not self.camera or not self.model:
            return []
        
        obstacles = []
        
        try:
            # Capture frame
            frame = self.camera.capture_array()
            if frame.shape[2] == 4:  # Remove alpha channel if present
                frame = frame[:, :, :3]
            
            # Run YOLO detection
            results = self.model.predict(frame, verbose=False, imgsz=self.config.camera_resolution[0])
            
            frame_height, frame_width = frame.shape[:2]
            
            for box in results[0].boxes:
                conf = float(box.conf)
                if conf < self.config.vision_conf_threshold:
                    continue
                
                cls_id = int(box.cls)
                cls_name = self.model.names[cls_id] if cls_id < len(self.model.names) else f"class{cls_id}"
                
                # Only process obstacle classes
                if cls_name not in self.obstacle_classes:
                    continue
                
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                
                # Calculate object center and size
                obj_center_x = (x1 + x2) / 2
                obj_center_y = (y1 + y2) / 2
                obj_width = x2 - x1
                obj_height = y2 - y1
                
                # Calculate angle from camera center
                angle_from_center = ((obj_center_x - frame_width/2) / frame_width) * self.config.camera_fov
                absolute_angle = current_heading + angle_from_center
                
                # Estimate distance based on object size (rough approximation)
                distance = self._estimate_distance_from_size(cls_name, obj_width, obj_height, frame_width, frame_height)
                
                # Calculate relative position
                angle_rad = math.radians(absolute_angle)
                rel_x = distance * math.sin(angle_rad)
                rel_y = distance * math.cos(angle_rad)
                
                # Estimate real-world size
                estimated_size = self._estimate_real_size(cls_name, distance)
                
                obstacle = DetectedObstacle(
                    type="vision",
                    object_class=cls_name,
                    confidence=conf,
                    distance=distance,
                    angle=angle_from_center,
                    position=(rel_x, rel_y),
                    timestamp=time.time(),
                    size=estimated_size
                )
                
                obstacles.append(obstacle)
                
        except Exception as e:
            print(f"Vision detection error: {e}")
        
        return obstacles
    
    def _estimate_distance_from_size(self, class_name: str, pixel_width: float, pixel_height: float, 
                                   frame_width: float, frame_height: float) -> float:
        """Estimate distance based on object size in pixels"""
        # Rough size estimates for common objects (in meters)
        object_sizes = {
            "person": (0.5, 1.7),  # width, height
            "chair": (0.5, 0.9),
            "table": (1.0, 0.8),
            "door": (0.8, 2.0),
            "stairs": (1.0, 0.2),
            "wall": (2.0, 2.5),
            "furniture": (0.8, 1.0),
            "backpack": (0.3, 0.4)
        }
        
        if class_name not in object_sizes:
            return 2.0  # Default distance
        
        real_width, real_height = object_sizes[class_name]
        
        # Use the larger dimension for more accurate estimation
        pixel_size = max(pixel_width, pixel_height)
        real_size = max(real_width, real_height)
        frame_size = max(frame_width, frame_height)
        
        # Simple distance estimation based on focal length approximation
        # This is a rough estimate and should be calibrated for better accuracy
        focal_length_approx = frame_size / (2 * math.tan(math.radians(self.config.camera_fov / 2)))
        distance = (real_size * focal_length_approx) / pixel_size
        
        return max(0.5, min(distance, self.config.max_obstacle_distance))
    
    def _estimate_real_size(self, class_name: str, distance: float) -> Tuple[float, float]:
        """Estimate real-world size of detected object"""
        size_estimates = {
            "person": (0.5, 1.7),
            "chair": (0.5, 0.9),
            "table": (1.0, 0.8),
            "door": (0.8, 2.0),
            "stairs": (1.0, 0.2),
            "wall": (2.0, 2.5),
            "furniture": (0.8, 1.0),
            "backpack": (0.3, 0.4)
        }
        
        return size_estimates.get(class_name, (0.5, 0.5))
    
    def stop(self):
        """Stop the vision detector"""
        if self.camera and self.owns_camera:
            self.camera.stop()
            self.camera = None
        self.is_running = False

class UltrasonicObstacleDetector:
    def __init__(self, config: ObstacleDetectionConfig):
        self.config = config
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize ultrasonic sensor"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.config.ultrasonic_trigger_pin, GPIO.OUT)
            GPIO.setup(self.config.ultrasonic_echo_pin, GPIO.IN)
            
            # Ensure trigger is low initially
            GPIO.output(self.config.ultrasonic_trigger_pin, GPIO.LOW)
            time.sleep(0.1)
            
            self.is_initialized = True
            print("Ultrasonic obstacle detector initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize ultrasonic detector: {e}")
            return False
    
    def detect_obstacles(self, current_heading: float = 0.0) -> List[DetectedObstacle]:
        """Detect obstacles using ultrasonic sensor"""
        if not self.is_initialized:
            return []
        
        obstacles = []
        
        try:
            # Trigger ultrasonic pulse
            GPIO.output(self.config.ultrasonic_trigger_pin, GPIO.HIGH)
            time.sleep(10e-6)  # 10 microseconds
            GPIO.output(self.config.ultrasonic_trigger_pin, GPIO.LOW)
            
            # Wait for echo start
            pulse_start = time.time()
            timeout = pulse_start + 0.1  # 100ms timeout
            
            while GPIO.input(self.config.ultrasonic_echo_pin) == GPIO.LOW:
                pulse_start = time.time()
                if pulse_start > timeout:
                    return obstacles  # Timeout, no echo
            
            # Wait for echo end
            pulse_end = time.time()
            timeout = pulse_end + 0.1
            
            while GPIO.input(self.config.ultrasonic_echo_pin) == GPIO.HIGH:
                pulse_end = time.time()
                if pulse_end > timeout:
                    return obstacles  # Timeout, echo stuck high
            
            # Calculate distance
            pulse_duration = pulse_end - pulse_start
            distance = (pulse_duration * 34300) / 2 / 100  # Convert to meters
            
            # Validate distance
            if (self.config.ultrasonic_min_distance <= distance <= self.config.ultrasonic_max_distance):
                # Calculate position (straight ahead)
                angle_rad = math.radians(current_heading)
                rel_x = distance * math.sin(angle_rad)
                rel_y = distance * math.cos(angle_rad)
                
                obstacle = DetectedObstacle(
                    type="ultrasonic",
                    object_class="obstacle",
                    confidence=0.9,  # High confidence for ultrasonic
                    distance=distance,
                    angle=0.0,  # Straight ahead
                    position=(rel_x, rel_y),
                    timestamp=time.time(),
                    size=(0.2, 0.2)  # Unknown size, assume small
                )
                
                obstacles.append(obstacle)
                
        except Exception as e:
            print(f"Ultrasonic detection error: {e}")
        
        return obstacles
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if self.is_initialized:
            GPIO.cleanup()
            self.is_initialized = False

class ObstacleDetectionSystem:
    def __init__(self, config: ObstacleDetectionConfig = None):
        self.config = config or ObstacleDetectionConfig()
        
        # Initialize detectors
        self.vision_detector = VisionObstacleDetector(self.config)
        self.ultrasonic_detector = UltrasonicObstacleDetector(self.config)
        
        # Detection state
        self.is_running = False
        self.detection_thread = None
        self.obstacle_history = deque(maxlen=100)
        self.current_obstacles = []
        self.obstacles_lock = threading.Lock()
        
        # TTS setup
        self.setup_tts()
        
        # Callbacks
        self.obstacle_callbacks = []
        self.path_blocking_callbacks = []
    
    def setup_tts(self):
        """Setup text-to-speech for obstacle warnings"""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_queue = queue.Queue()
            
            # Start TTS worker thread
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
            
        except Exception as e:
            print(f"TTS setup failed: {e}")
            self.tts_engine = None
    
    def _tts_worker(self):
        """TTS worker thread"""
        while True:
            try:
                msg = self.tts_queue.get(timeout=1)
                if msg is None:
                    break
                if self.tts_engine:
                    self.tts_engine.say(msg)
                    self.tts_engine.runAndWait()
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS error: {e}")
    
    def initialize(self) -> bool:
        """Initialize the obstacle detection system"""
        vision_ok = self.vision_detector.initialize()
        ultrasonic_ok = self.ultrasonic_detector.initialize()
        
        if not vision_ok and not ultrasonic_ok:
            print("Failed to initialize any obstacle detectors")
            return False
        
        if not vision_ok:
            print("Warning: Vision detector failed to initialize")
        if not ultrasonic_ok:
            print("Warning: Ultrasonic detector failed to initialize")
        
        print("Obstacle detection system initialized")
        return True
    
    def start_detection(self, current_heading_callback=None):
        """Start obstacle detection"""
        if self.is_running:
            return
        
        self.is_running = True
        self.current_heading_callback = current_heading_callback
        
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        print("Obstacle detection started")
    
    def stop_detection(self):
        """Stop obstacle detection"""
        self.is_running = False
        
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        self.vision_detector.stop()
        self.ultrasonic_detector.cleanup()
        
        if hasattr(self, 'tts_queue'):
            self.tts_queue.put(None)
        
        print("Obstacle detection stopped")
    
    def _detection_loop(self):
        """Main detection loop"""
        last_detection = 0
        detection_interval = 1.0 / self.config.detection_rate
        
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - last_detection >= detection_interval:
                    # Get current heading
                    current_heading = 0.0
                    if self.current_heading_callback:
                        try:
                            current_heading = self.current_heading_callback()
                        except:
                            pass
                    
                    # Detect obstacles
                    all_obstacles = []
                    
                    # Vision detection
                    vision_obstacles = self.vision_detector.detect_obstacles(current_heading)
                    all_obstacles.extend(vision_obstacles)
                    
                    # Ultrasonic detection
                    ultrasonic_obstacles = self.ultrasonic_detector.detect_obstacles(current_heading)
                    all_obstacles.extend(ultrasonic_obstacles)
                    
                    # Filter and process obstacles
                    filtered_obstacles = self._filter_obstacles(all_obstacles)
                    
                    # Update current obstacles
                    with self.obstacles_lock:
                        self.current_obstacles = filtered_obstacles
                        self.obstacle_history.extend(filtered_obstacles)
                    
                    # Process obstacles
                    self._process_obstacles(filtered_obstacles)
                    
                    last_detection = current_time
                
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                
            except Exception as e:
                print(f"Detection loop error: {e}")
                time.sleep(0.1)
    
    def _filter_obstacles(self, obstacles: List[DetectedObstacle]) -> List[DetectedObstacle]:
        """Filter and validate detected obstacles"""
        filtered = []
        
        for obstacle in obstacles:
            # Distance validation
            if obstacle.distance > self.config.max_obstacle_distance:
                continue
            
            # Size validation
            if max(obstacle.size) < self.config.min_obstacle_size:
                continue
            
            # Confidence validation
            if obstacle.confidence < 0.3:
                continue
            
            filtered.append(obstacle)
        
        return filtered
    
    def _process_obstacles(self, obstacles: List[DetectedObstacle]):
        """Process detected obstacles and trigger callbacks"""
        if not obstacles:
            return
        
        # Find critical obstacles
        critical_obstacles = [obs for obs in obstacles if obs.distance <= self.config.critical_distance]
        warning_obstacles = [obs for obs in obstacles if obs.distance <= self.config.warning_distance]
        
        # Trigger obstacle callbacks
        for callback in self.obstacle_callbacks:
            try:
                callback(obstacles, critical_obstacles, warning_obstacles)
            except Exception as e:
                print(f"Obstacle callback error: {e}")
        
        # Check for path blocking obstacles
        blocking_obstacles = self._find_path_blocking_obstacles(obstacles)
        if blocking_obstacles:
            for callback in self.path_blocking_callbacks:
                try:
                    callback(blocking_obstacles)
                except Exception as e:
                    print(f"Path blocking callback error: {e}")
        
        # TTS warnings
        self._announce_obstacles(critical_obstacles, warning_obstacles)
    
    def _find_path_blocking_obstacles(self, obstacles: List[DetectedObstacle]) -> List[DetectedObstacle]:
        """Find obstacles that might block the current path"""
        blocking = []
        
        for obstacle in obstacles:
            # Check if obstacle is in front and close enough to block path
            if (abs(obstacle.angle) < 30 and  # Within 30 degrees of center
                obstacle.distance < self.config.path_blocking_threshold):
                blocking.append(obstacle)
        
        return blocking
    
    def _announce_obstacles(self, critical_obstacles: List[DetectedObstacle], warning_obstacles: List[DetectedObstacle]):
        """Announce obstacles via TTS"""
        if not hasattr(self, 'tts_queue') or not self.tts_engine:
            return
        
        current_time = time.time()
        
        # Announce critical obstacles immediately
        for obstacle in critical_obstacles:
            if not hasattr(self, 'last_critical_announcement') or current_time - self.last_critical_announcement > 2.0:
                msg = f"Critical obstacle: {obstacle.object_class} at {obstacle.distance:.1f} meters"
                self.tts_queue.put(msg)
                self.last_critical_announcement = current_time
                break
        
        # Announce warning obstacles with longer cooldown
        if warning_obstacles and (not hasattr(self, 'last_warning_announcement') or current_time - self.last_warning_announcement > 5.0):
            closest = min(warning_obstacles, key=lambda x: x.distance)
            msg = f"Obstacle ahead: {closest.object_class} at {closest.distance:.1f} meters"
            self.tts_queue.put(msg)
            self.last_warning_announcement = current_time
    
    def get_current_obstacles(self) -> List[DetectedObstacle]:
        """Get current obstacles (thread-safe)"""
        with self.obstacles_lock:
            return self.current_obstacles.copy()
    
    def get_obstacles_in_area(self, center: Tuple[float, float], radius: float) -> List[DetectedObstacle]:
        """Get obstacles within a specific area"""
        obstacles_in_area = []
        
        with self.obstacles_lock:
            for obstacle in self.current_obstacles:
                distance_to_center = math.sqrt(
                    (obstacle.position[0] - center[0])**2 + 
                    (obstacle.position[1] - center[1])**2
                )
                
                if distance_to_center <= radius:
                    obstacles_in_area.append(obstacle)
        
        return obstacles_in_area
    
    def add_obstacle_callback(self, callback):
        """Add callback for obstacle detection events"""
        self.obstacle_callbacks.append(callback)
    
    def add_path_blocking_callback(self, callback):
        """Add callback for path blocking events"""
        self.path_blocking_callbacks.append(callback)
    
    def is_path_clear(self, start: Tuple[float, float], end: Tuple[float, float], 
                     path_width: float = 0.6) -> bool:
        """Check if a path is clear of obstacles"""
        with self.obstacles_lock:
            for obstacle in self.current_obstacles:
                # Calculate distance from obstacle to line segment
                distance_to_path = self._point_to_line_distance(
                    obstacle.position, start, end
                )
                
                # Consider obstacle size
                obstacle_radius = max(obstacle.size) / 2
                
                if distance_to_path < (path_width / 2 + obstacle_radius):
                    return False
        
        return True
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                               line_start: Tuple[float, float], 
                               line_end: Tuple[float, float]) -> float:
        """Calculate distance from point to line segment"""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate line length
        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if line_length == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Calculate projection parameter
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_length**2)))
        
        # Calculate closest point on line
        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)
        
        # Return distance
        return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

# Example usage and testing functions
def test_obstacle_detection():
    """Test function for obstacle detection system"""
    config = ObstacleDetectionConfig()
    detector = ObstacleDetectionSystem(config)
    
    if not detector.initialize():
        print("Failed to initialize obstacle detection system")
        return
    
    def obstacle_callback(obstacles, critical, warning):
        print(f"Detected {len(obstacles)} obstacles:")
        for obs in obstacles:
            print(f"  {obs.object_class}: {obs.distance:.2f}m at {obs.angle:.1f}°")
    
    def path_blocking_callback(blocking_obstacles):
        print(f"Path blocked by {len(blocking_obstacles)} obstacles!")
        for obs in blocking_obstacles:
            print(f"  Blocking: {obs.object_class} at {obs.distance:.2f}m")
    
    detector.add_obstacle_callback(obstacle_callback)
    detector.add_path_blocking_callback(path_blocking_callback)
    
    try:
        detector.start_detection()
        
        print("Obstacle detection running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
            
            # Test path clearance
            obstacles = detector.get_current_obstacles()
            if obstacles:
                is_clear = detector.is_path_clear((0, 0), (0, 2))
                print(f"Path ahead clear: {is_clear}")
    
    except KeyboardInterrupt:
        print("Stopping obstacle detection...")
    
    finally:
        detector.stop_detection()

if __name__ == "__main__":
    test_obstacle_detection()