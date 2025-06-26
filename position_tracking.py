import os
import sys
# Fix Qt platform plugin issues BEFORE importing other modules
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Use headless Qt
# Alternative: os.environ['MPLBACKEND'] = 'Agg'  # Use non-interactive matplotlib backend

from datetime import datetime
import board
import busio
import smbus
import numpy as np
import matplotlib
# Set matplotlib backend before importing pyplot
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have Qt5, or 'Agg' for headless
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import cv2
import dt_apriltags as apriltag
from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont

# Set OpenCV to not use Qt
cv2.setUseOptimized(True)

# MPU6050 registers
MPU6050_ADDR = 0x68
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
PWR_MGMT_1 = 0x6B
INT_ENABLE = 0x38
INT_BYPASS_EN = 0x37

# QMC5883L registers
QMC5883L_ADDR = 0x0D
QMC_REG_CONTROL = 0x09
QMC_REG_RESET = 0x0B
QMC_DATA_REGISTER = 0x00

# BMP180 registers
BMP180_ADDR = 0x77
BMP180_CONTROL = 0xF4
BMP180_DATA = 0xF6
BMP180_CAL_AC1 = 0xAA

@dataclass
class IMUData:
    """Container for IMU sensor readings"""
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0
    timestamp: float = 0.0

@dataclass
class Position2D:
    """Container for 2D position and velocity"""
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    latitude: float = 36.7048  # Initial latitude
    longitude: float = 3.1745  # Initial longitude

@dataclass
class AprilTagDetection:
    """Container for AprilTag detection data"""
    tag_id: int
    center: Tuple[float, float]
    corners: np.ndarray
    pose_translation: Optional[np.ndarray] = None
    pose_rotation: Optional[np.ndarray] = None
    confidence: float = 0.0

class LowPassFilter:
    """Simple low-pass filter for noise reduction"""
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False
    
    def update(self, new_value: float) -> float:
        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

class ComplementaryFilter:
    """Complementary filter for attitude estimation (98% gyro, 2% accel)"""
    
    def __init__(self, alpha: float = 0.98, dt: float = 0.02):
        self.alpha = alpha  # 98% gyroscope, 2% accelerometer
        self.dt = dt
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
    
    def update(self, accel: np.ndarray, gyro: np.ndarray, mag: Optional[np.ndarray] = None):
        """Update attitude estimates using complementary filter"""
        # Convert gyro from degrees to radians
        gyro_rad = np.radians(gyro)
        
        # Calculate angles from accelerometer
        accel_roll = math.atan2(accel[1], accel[2])
        accel_pitch = math.atan2(-accel[0], math.sqrt(accel[1]**2 + accel[2]**2))
        
        # Integrate gyroscope (98% weight)
        self.roll += gyro_rad[0] * self.dt
        self.pitch += gyro_rad[1] * self.dt
        self.yaw += gyro_rad[2] * self.dt
        
        # Apply complementary filter (2% accelerometer correction)
        self.roll = self.alpha * self.roll + (1 - self.alpha) * accel_roll
        self.pitch = self.alpha * self.pitch + (1 - self.alpha) * accel_pitch
        
        # Use magnetometer for yaw correction if available
        if mag is not None:
            try:
                # Calculate magnetic heading with tilt compensation
                mag_x_comp = mag[0] * math.cos(self.pitch) + mag[2] * math.sin(self.pitch)
                mag_y_comp = mag[0] * math.sin(self.roll) * math.sin(self.pitch) + \
                            mag[1] * math.cos(self.roll) - \
                            mag[2] * math.sin(self.roll) * math.cos(self.pitch)
                
                mag_yaw = math.atan2(-mag_y_comp, mag_x_comp)
                
                # Apply weak correction to yaw (10% magnetometer influence)
                yaw_diff = mag_yaw - self.yaw
                # Handle angle wrap-around
                if yaw_diff > math.pi:
                    yaw_diff -= 2 * math.pi
                elif yaw_diff < -math.pi:
                    yaw_diff += 2 * math.pi
                
                self.yaw = self.yaw + 0.1 * yaw_diff
                
            except Exception as e:
                print(f"Magnetometer fusion error: {e}")
        
        return self.roll, self.pitch, self.yaw

class AprilTagTracker:
    """AprilTag detection and tracking for position correction"""
    
    def __init__(self, camera_resolution: Tuple[int, int] = (640, 480), shared_camera=None):
        self.detector = apriltag.Detector(families='tag36h11')
        self.camera_resolution = camera_resolution
        
        # Initialize camera - use shared camera if provided
        self.picam2 = shared_camera
        self.owns_camera = shared_camera is None  # Track if we own the camera
        
        if self.owns_camera:
            try:
                self.picam2 = Picamera2()
                self.picam2.preview_configuration.main.size = camera_resolution
                self.picam2.preview_configuration.main.format = "RGB888"
                self.picam2.configure("preview")
                self.picam2.start()
                print("AprilTagTracker: Camera initialized successfully")
            except Exception as e:
                print(f"AprilTagTracker: Camera initialization failed: {e}")
                self.picam2 = None
        else:
            print("AprilTagTracker: Using shared camera instance")
        
        # AprilTag database (tag_id -> real world position in meters)
        self.tag_positions = {
            0: (0.0, 0.0),      # Origin tag
            1: (2.0, 0.0),      # 2 meters along X
            2: (0.0, 2.0),      # 2 meters along Y
            3: (2.0, 2.0),      # Corner tag
            4: (-2.0, 0.0),     # Negative X
            5: (0.0, -2.0),     # Negative Y
        }
        
        # Camera calibration parameters (you should calibrate your camera)
        self.camera_matrix = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Tag size in meters (measure your actual tags)
        self.tag_size = 0.1  # 10cm tags
        
        # Detection parameters
        self.detection_lock = threading.Lock()
        self.latest_detections = []
        self.detection_confidence_threshold = 0.5
        
        # Position correction parameters
        self.position_corrections = deque(maxlen=100)
        self.last_correction_time = 0
        self.correction_cooldown = 0.5  # seconds between corrections
        
    def detect_tags(self) -> list:
        """Detect AprilTags in current camera frame"""
        if not self.picam2:
            return []
        
        try:
            # Capture frame
            frame = self.picam2.capture_array()
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detect AprilTags
            detections = self.detector.detect(gray)
            
            processed_detections = []
            
            for detection in detections:
                # Calculate pose if tag is in our database
                if detection.tag_id in self.tag_positions:
                    try:
                        # Estimate pose
                        object_points = np.array([
                            [-self.tag_size/2, -self.tag_size/2, 0],
                            [ self.tag_size/2, -self.tag_size/2, 0],
                            [ self.tag_size/2,  self.tag_size/2, 0],
                            [-self.tag_size/2,  self.tag_size/2, 0]
                        ], dtype=np.float32)
                        
                        image_points = detection.corners.astype(np.float32)
                        
                        success, rvec, tvec = cv2.solvePnP(
                            object_points, image_points, 
                            self.camera_matrix, self.dist_coeffs
                        )
                        
                        if success:
                            # Calculate confidence based on detection quality
                            confidence = min(1.0, detection.decision_margin / 50.0)
                            
                            processed_detection = AprilTagDetection(
                                tag_id=detection.tag_id,
                                center=tuple(detection.center),
                                corners=detection.corners,
                                pose_translation=tvec.flatten(),
                                pose_rotation=rvec.flatten(),
                                confidence=confidence
                            )
                            processed_detections.append(processed_detection)
                            
                    except Exception as e:
                        print(f"Pose estimation error for tag {detection.tag_id}: {e}")
                        # Still add detection without pose
                        processed_detection = AprilTagDetection(
                            tag_id=detection.tag_id,
                            center=tuple(detection.center),
                            corners=detection.corners,
                            confidence=0.3
                        )
                        processed_detections.append(processed_detection)
            
            with self.detection_lock:
                self.latest_detections = processed_detections
            
            return processed_detections
            
        except Exception as e:
            print(f"AprilTag detection error: {e}")
            return []
    
    def calculate_position_from_tags(self, detections: list) -> Optional[Tuple[float, float, float]]:
        """Calculate robot position from AprilTag detections"""
        if not detections:
            return None
        
        # Use the most confident detection
        best_detection = max(detections, key=lambda d: d.confidence)
        
        if best_detection.confidence < self.detection_confidence_threshold:
            return None
        
        if best_detection.tag_id not in self.tag_positions:
            return None
        
        # Get tag's real-world position
        tag_world_x, tag_world_y = self.tag_positions[best_detection.tag_id]
        
        if best_detection.pose_translation is not None:
            # Use 3D pose estimation
            # Transform camera-relative position to world coordinates
            cam_x, cam_y, cam_z = best_detection.pose_translation
            
            # Simple transformation (assumes camera is level and facing forward)
            # You might need to adjust this based on your camera mounting
            robot_x = tag_world_x - cam_z  # Camera Z is forward distance
            robot_y = tag_world_y + cam_x  # Camera X is lateral offset
            
            return robot_x, robot_y, best_detection.confidence
        
        else:
            # Use simpler 2D estimation based on image coordinates
            # This is less accurate but more robust
            center_x, center_y = best_detection.center
            image_center_x = self.camera_resolution[0] / 2
            image_center_y = self.camera_resolution[1] / 2
            
            # Simple proportional estimation (you should calibrate this)
            pixel_to_meter = 0.01  # Adjust based on your setup
            
            offset_x = (center_x - image_center_x) * pixel_to_meter
            offset_y = (center_y - image_center_y) * pixel_to_meter
            
            robot_x = tag_world_x - offset_x
            robot_y = tag_world_y - offset_y
            
            return robot_x, robot_y, best_detection.confidence * 0.5  # Lower confidence for 2D
    
    def get_latest_detections(self) -> list:
        """Get latest AprilTag detections"""
        with self.detection_lock:
            return self.latest_detections.copy()
    
    def stop(self):
        """Stop camera and cleanup"""
        if self.picam2 and self.owns_camera:
            self.picam2.stop()
            print("AprilTagTracker: Camera stopped")

class EnhancedIMUTracker:
    """Enhanced IMU tracker with complementary filter, drift correction, and AprilTag integration"""
    
    def __init__(self, sample_rate: float = 50.0, enable_apriltag: bool = True):
        self.dt = 1.0 / sample_rate
        self.position = Position2D()
        
        # Initialize I2C
        self.bus = smbus.SMBus(1)
        
        # Complementary filter
        self.complementary_filter = ComplementaryFilter(alpha=0.98, dt=self.dt)
        
        # Low-pass filters for acceleration
        self.accel_filters = [LowPassFilter(alpha=0.1) for _ in range(3)]
        
        # Calibration offsets
        self.accel_offset = np.array([0.0, 0.0, 0.0])
        self.gyro_offset = np.array([0.0, 0.0, 0.0])
        self.mag_offset = np.array([0.0, 0.0, 0.0])
        self.mag_scale = np.array([1.0, 1.0, 1.0])
        
        # Scale factors
        self.accel_scale = 8192.0   # ±4g range
        self.gyro_scale = 65.5      # ±500°/s range
        
        # Data storage
        self.position_history = deque(maxlen=1000)
        self.running = False
        
        # Drift correction parameters
        self.drift_correction_enabled = True
        self.last_zero_velocity_time = time.time()
        self.velocity_threshold = 0.05  # m/s
        self.stationary_time_threshold = 1.0  # seconds
        
        # Path tracking
        self.path_coords = []
        self.path_lock = threading.Lock()
        
        # AprilTag integration
        self.apriltag_enabled = enable_apriltag
        if self.apriltag_enabled:
            self.apriltag_tracker = AprilTagTracker()
            self.last_apriltag_correction = 0
            self.apriltag_correction_interval = 1.0  # seconds
        else:
            self.apriltag_tracker = None
        
        # Initialization flag
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the IMU tracker and sensors"""
        try:
            self.initialize_sensors()
            self.initialized = True
            print("EnhancedIMUTracker initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize EnhancedIMUTracker: {e}")
            return False
    
    def read_word_data(self, addr: int, reg: int) -> int:
        """Read 16-bit unsigned data"""
        high = self.bus.read_byte_data(addr, reg)
        low = self.bus.read_byte_data(addr, reg + 1)
        return (high << 8) | low
    
    def read_word_data_signed(self, addr: int, reg: int) -> int:
        """Read 16-bit signed data"""
        value = self.read_word_data(addr, reg)
        if value > 32767:
            value -= 65536
        return value
    
    def initialize_sensors(self):
        """Initialize all sensors"""
        try:
            # Initialize MPU6050
            self.bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)
            time.sleep(0.1)
            self.bus.write_byte_data(MPU6050_ADDR, 0x1C, 0x08)  # ±4g
            self.bus.write_byte_data(MPU6050_ADDR, 0x1B, 0x08)  # ±500°/s
            self.bus.write_byte_data(MPU6050_ADDR, INT_BYPASS_EN, 0x02)
            print("MPU6050 initialized")
            
            # Initialize QMC5883L
            self.bus.write_byte_data(QMC5883L_ADDR, QMC_REG_RESET, 0x01)
            time.sleep(0.1)
            self.bus.write_byte_data(QMC5883L_ADDR, QMC_REG_CONTROL, 0x1D)
            time.sleep(0.1)
            print("QMC5883L initialized")
            
        except Exception as e:
            print(f"Sensor initialization error: {e}")
            raise
    
    def read_mpu6050(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read MPU6050 accelerometer and gyroscope data"""
        try:
            # Read accelerometer data
            accel_x = self.read_word_data_signed(MPU6050_ADDR, ACCEL_XOUT_H) / self.accel_scale
            accel_y = self.read_word_data_signed(MPU6050_ADDR, ACCEL_XOUT_H + 2) / self.accel_scale
            accel_z = self.read_word_data_signed(MPU6050_ADDR, ACCEL_XOUT_H + 4) / self.accel_scale
            
            # Read gyroscope data
            gyro_x = self.read_word_data_signed(MPU6050_ADDR, GYRO_XOUT_H) / self.gyro_scale
            gyro_y = self.read_word_data_signed(MPU6050_ADDR, GYRO_XOUT_H + 2) / self.gyro_scale
            gyro_z = self.read_word_data_signed(MPU6050_ADDR, GYRO_XOUT_H + 4) / self.gyro_scale
            
            # Apply calibration
            accel = np.array([accel_x, accel_y, accel_z]) - self.accel_offset
            gyro = np.array([gyro_x, gyro_y, gyro_z]) - self.gyro_offset
            
            return accel, gyro
            
        except Exception as e:
            print(f"MPU6050 read error: {e}")
            return np.zeros(3), np.zeros(3)
    
    def read_qmc5883l(self) -> np.ndarray:
        """Read QMC5883L magnetometer data"""
        try:
            # Read 6 bytes of data
            data = self.bus.read_i2c_block_data(QMC5883L_ADDR, QMC_DATA_REGISTER, 6)
            
            # Convert to signed 16-bit values
            mag_x = (data[1] << 8) | data[0]
            mag_y = (data[3] << 8) | data[2]
            mag_z = (data[5] << 8) | data[4]
            
            # Convert to signed values
            if mag_x > 32767: mag_x -= 65536
            if mag_y > 32767: mag_y -= 65536
            if mag_z > 32767: mag_z -= 65536
            
            # Apply calibration
            mag = np.array([mag_x, mag_y, mag_z], dtype=float) - self.mag_offset
            mag = mag * self.mag_scale
            
            return mag
            
        except Exception as e:
            print(f"QMC5883L read error: {e}")
            return np.zeros(3)
    
    def transform_acceleration(self, accel: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Transform acceleration from body frame to world frame using full 3D rotation matrix"""
        # Create rotation matrix from body to world frame (ZYX Euler angles)
        cos_roll, sin_roll = math.cos(roll), math.sin(roll)
        cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        
        # Full 3D rotation matrix
        R = np.array([
            [cos_yaw * cos_pitch,
             cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
             cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll],
            [sin_yaw * cos_pitch,
             sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
             sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll],
            [-sin_pitch,
             cos_pitch * sin_roll,
             cos_pitch * cos_roll]
        ])
        
        # Transform acceleration to world frame
        world_accel = R @ accel
        
        # Remove gravity (assuming Z is up)
        world_accel[2] -= 1.0  # 1g in our units
        
        return world_accel
    
    def apply_drift_correction(self):
        """Apply velocity decay when stationary for drift correction"""
        if not self.drift_correction_enabled:
            return
        
        # Calculate velocity magnitude
        velocity_magnitude = math.sqrt(self.position.vx**2 + self.position.vy**2)
        
        # Check if stationary
        if velocity_magnitude < self.velocity_threshold:
            current_time = time.time()
            if current_time - self.last_zero_velocity_time > self.stationary_time_threshold:
                # Apply exponential velocity decay
                decay_factor = 0.95
                self.position.vx *= decay_factor
                self.position.vy *= decay_factor
        else:
            self.last_zero_velocity_time = time.time()
    
    def apply_apriltag_correction(self):
        """Apply position correction using AprilTag detections"""
        if not self.apriltag_enabled or not self.apriltag_tracker:
            return
        
        current_time = time.time()
        if current_time - self.last_apriltag_correction < self.apriltag_correction_interval:
            return
        
        try:
            # Detect AprilTags
            detections = self.apriltag_tracker.detect_tags()
            
            if detections:
                # Calculate position from tags
                tag_position = self.apriltag_tracker.calculate_position_from_tags(detections)
                
                if tag_position:
                    tag_x, tag_y, confidence = tag_position
                    
                    # Apply correction with confidence weighting
                    correction_factor = min(0.3, confidence * 0.5)  # Max 30% correction
                    
                    error_x = tag_x - self.position.x
                    error_y = tag_y - self.position.y
                    
                    self.position.x += error_x * correction_factor
                    self.position.y += error_y * correction_factor
                    
                    # Also apply some velocity correction to reduce drift
                    self.position.vx *= (1.0 - correction_factor * 0.2)
                    self.position.vy *= (1.0 - correction_factor * 0.2)
                    
                    print(f"AprilTag correction: ({error_x:.2f}, {error_y:.2f}) confidence: {confidence:.2f}")
                    
                    self.last_apriltag_correction = current_time
            
        except Exception as e:
            print(f"AprilTag correction error: {e}")
    
    def update_position(self):
        """Update position estimate using sensor fusion and coordinate transformation"""
        try:
            # Read sensor data
            accel, gyro = self.read_mpu6050()
            mag = self.read_qmc5883l()
            
            # Apply low-pass filtering to acceleration
            filtered_accel = np.array([
                self.accel_filters[i].update(accel[i]) for i in range(3)
            ])
            
            # Update attitude using complementary filter (98% gyro, 2% accel)
            roll, pitch, yaw = self.complementary_filter.update(filtered_accel, gyro, mag)
            
            # Transform acceleration to world coordinates using full 3D rotation matrix
            world_accel = self.transform_acceleration(filtered_accel, roll, pitch, yaw)
            
            # Extract horizontal acceleration (X, Y)
            accel_x, accel_y = world_accel[0], world_accel[1]
            
            # Apply dead zone to reduce noise
            dead_zone = 0.1
            if abs(accel_x) < dead_zone:
                accel_x = 0.0
            if abs(accel_y) < dead_zone:
                accel_y = 0.0
            
            # Convert acceleration to m/s² (assuming 1g = 9.81 m/s²)
            accel_x *= 9.81
            accel_y *= 9.81
            
            # Integrate acceleration to velocity
            self.position.vx += accel_x * self.dt
            self.position.vy += accel_y * self.dt
            
            # Integrate velocity to position
            delta_x = self.position.vx * self.dt
            delta_y = self.position.vy * self.dt
            
            self.position.x += delta_x
            self.position.y += delta_y
            
            # Store attitude
            self.position.roll = roll
            self.position.pitch = pitch
            self.position.yaw = yaw
            
            # Apply drift correction
            self.apply_drift_correction()
            
            # Apply AprilTag correction
            self.apply_apriltag_correction()
            
            # Convert to GPS coordinates (simple approximation)
            # 1 meter ≈ 9e-6 degrees latitude, 1 meter ≈ 9e-6/cos(lat) degrees longitude
            lat_per_meter = 9e-6
            lon_per_meter = 9e-6 / math.cos(math.radians(self.position.latitude))
            
            self.position.latitude += delta_y * lat_per_meter
            self.position.longitude += delta_x * lon_per_meter
            
            # Store position for visualization
            self.position_history.append((self.position.x, self.position.y, time.time()))
            
            # Update path coordinates
            with self.path_lock:
                self.path_coords.append((self.position.latitude, self.position.longitude))
                # Keep only last 1000 points
                if len(self.path_coords) > 1000:
                    self.path_coords.pop(0)
            
        except Exception as e:
            print(f"Position update error: {e}")
    
    def calibrate(self) -> bool:
        """Calibrate IMU sensors - wrapper for calibrate_sensors"""
        try:
            self.calibrate_sensors()
            return True
        except Exception as e:
            print(f"Calibration failed: {e}")
            return False
    
    def calibrate_sensors(self, samples: int = 1000, duration: float = 10.0):
        """Calibrate sensors by computing bias offsets"""
        print(f"Calibrating sensors... Keep IMU stationary for {duration} seconds")
        
        accel_sum = np.zeros(3)
        gyro_sum = np.zeros(3)
        mag_sum = np.zeros(3)
        
        start_time = time.time()
        count = 0
        
        while count < samples and (time.time() - start_time) < duration:
            try:
                # Read raw data
                accel_x = self.read_word_data_signed(MPU6050_ADDR, ACCEL_XOUT_H) / self.accel_scale
                accel_y = self.read_word_data_signed(MPU6050_ADDR, ACCEL_XOUT_H + 2) / self.accel_scale
                accel_z = self.read_word_data_signed(MPU6050_ADDR, ACCEL_XOUT_H + 4) / self.accel_scale
                
                gyro_x = self.read_word_data_signed(MPU6050_ADDR, GYRO_XOUT_H) / self.gyro_scale
                gyro_y = self.read_word_data_signed(MPU6050_ADDR, GYRO_XOUT_H + 2) / self.gyro_scale
                gyro_z = self.read_word_data_signed(MPU6050_ADDR, GYRO_XOUT_H + 4) / self.gyro_scale
                
                mag = self.read_qmc5883l()
                
                accel_sum += np.array([accel_x, accel_y, accel_z])
                gyro_sum += np.array([gyro_x, gyro_y, gyro_z])
                mag_sum += mag
                
                count += 1
                time.sleep(0.02)  # 50Hz
                
            except Exception as e:
                print(f"Calibration error: {e}")
                continue
        
        # Calculate offsets
        self.accel_offset = accel_sum / count
        self.gyro_offset = gyro_sum / count
        self.mag_offset = mag_sum / count
        
        # Adjust Z-axis accel offset to account for gravity
        self.accel_offset[2] -= 1.0  # Remove gravity
        
        print(f"Calibration complete with {count} samples")
        print(f"Accel offsets: {self.accel_offset}")
        print(f"Gyro offsets: {self.gyro_offset}")
        print(f"Mag offsets: {self.mag_offset}")

    def reset_position(self, x: float = 0.0, y: float = 0.0, lat: float = 36.7048, lon: float = 3.1745):
        """Reset position to specified coordinates"""
        self.position.x = x
        self.position.y = y
        self.position.vx = 0.0
        self.position.vy = 0.0
        self.position.latitude = lat
        self.position.longitude = lon
        
        # Clear history
        self.position_history.clear()
        with self.path_lock:
            self.path_coords.clear()
        
        print(f"Position reset to ({x:.2f}, {y:.2f})")
    
    def start(self) -> bool:
        """Start position tracking - wrapper for start_tracking"""
        if not self.initialized:
            print("Tracker not initialized. Call initialize() first.")
            return False
        
        try:
            self.start_tracking()
            return True
        except Exception as e:
            print(f"Failed to start tracking: {e}")
            return False
    
    def start_tracking(self):
        """Start continuous position tracking"""
        self.running = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        print("Position tracking started")
    
    def stop(self):
        """Stop position tracking - wrapper for stop_tracking"""
        self.stop_tracking()
    
    def stop_tracking(self):
        """Stop position tracking"""
        self.running = False
        if hasattr(self, 'tracking_thread'):
            self.tracking_thread.join(timeout=1.0)
        
        if self.apriltag_tracker:
            self.apriltag_tracker.stop()
        
        print("Position tracking stopped")
    
    def _tracking_loop(self):
        """Main tracking loop running at specified sample rate"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Maintain consistent sample rate
            if current_time - last_time >= self.dt:
                self.update_position()
                last_time = current_time
            
            time.sleep(0.001)  # Small sleep to prevent CPU overload
    
    def get_current_position(self) -> Position2D:
        """Get current position estimate"""
        return self.position
    
    def get_position_history(self) -> list:
        """Get position history for visualization"""
        return list(self.position_history)
    
    def get_path_coordinates(self) -> list:
        """Get GPS path coordinates"""
        with self.path_lock:
            return self.path_coords.copy()
    
    def print_status(self):
        """Print current status information"""
        print(f"Position: ({self.position.x:.2f}, {self.position.y:.2f}) m")
        print(f"Velocity: ({self.position.vx:.2f}, {self.position.vy:.2f}) m/s")
        print(f"Attitude: Roll={math.degrees(self.position.roll):.1f}°, "
              f"Pitch={math.degrees(self.position.pitch):.1f}°, "
              f"Yaw={math.degrees(self.position.yaw):.1f}°")
        print(f"GPS: ({self.position.latitude:.6f}, {self.position.longitude:.6f})")
        
        if self.apriltag_tracker:
            detections = self.apriltag_tracker.get_latest_detections()
            print(f"AprilTags detected: {len(detections)}")


class PositionVisualizer:
    """Real-time position visualization using matplotlib"""
    
    def __init__(self, tracker: EnhancedIMUTracker, update_interval: int = 100):
        self.tracker = tracker
        self.update_interval = update_interval
        
        # Setup plot
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle('IMU Position Tracking System', fontsize=16)
        
        # Position plot
        self.ax1.set_title('2D Position Tracking')
        self.ax1.set_xlabel('X (meters)')
        self.ax1.set_ylabel('Y (meters)')
        self.ax1.grid(True)
        self.ax1.set_aspect('equal')
        
        self.position_line, = self.ax1.plot([], [], 'b-', alpha=0.7, linewidth=2, label='Path')
        self.current_pos, = self.ax1.plot([], [], 'ro', markersize=8, label='Current')
        self.ax1.legend()
        
        # Velocity plot
        self.ax2.set_title('Velocity Components')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Velocity (m/s)')
        self.ax2.grid(True)
        
        self.vel_x_line, = self.ax2.plot([], [], 'r-', label='Vx')
        self.vel_y_line, = self.ax2.plot([], [], 'g-', label='Vy')
        self.ax2.legend()
        
        # Attitude plot
        self.ax3.set_title('Attitude (Euler Angles)')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Angle (degrees)')
        self.ax3.grid(True)
        
        self.roll_line, = self.ax3.plot([], [], 'r-', label='Roll')
        self.pitch_line, = self.ax3.plot([], [], 'g-', label='Pitch')
        self.yaw_line, = self.ax3.plot([], [], 'b-', label='Yaw')
        self.ax3.legend()
        
        # GPS coordinates plot
        self.ax4.set_title('GPS Path')
        self.ax4.set_xlabel('Longitude')
        self.ax4.set_ylabel('Latitude')
        self.ax4.grid(True)
        
        self.gps_line, = self.ax4.plot([], [], 'g-', alpha=0.7, linewidth=2, label='GPS Path')
        self.gps_current, = self.ax4.plot([], [], 'go', markersize=6, label='Current GPS')
        self.ax4.legend()
        
        # Data storage for plots
        self.time_data = deque(maxlen=500)
        self.pos_x_data = deque(maxlen=500)
        self.pos_y_data = deque(maxlen=500)
        self.vel_x_data = deque(maxlen=500)
        self.vel_y_data = deque(maxlen=500)
        self.roll_data = deque(maxlen=500)
        self.pitch_data = deque(maxlen=500)
        self.yaw_data = deque(maxlen=500)
        
        self.start_time = time.time()
    
    def update_plot(self, frame):
        """Update all plots with latest data"""
        try:
            current_time = time.time() - self.start_time
            position = self.tracker.get_current_position()
            
            # Store data
            self.time_data.append(current_time)
            self.pos_x_data.append(position.x)
            self.pos_y_data.append(position.y)
            self.vel_x_data.append(position.vx)
            self.vel_y_data.append(position.vy)
            self.roll_data.append(math.degrees(position.roll))
            self.pitch_data.append(math.degrees(position.pitch))
            self.yaw_data.append(math.degrees(position.yaw))
            
            # Update position plot
            self.position_line.set_data(list(self.pos_x_data), list(self.pos_y_data))
            self.current_pos.set_data([position.x], [position.y])
            
            # Auto-scale position plot
            if self.pos_x_data and self.pos_y_data:
                x_margin = max(1.0, (max(self.pos_x_data) - min(self.pos_x_data)) * 0.1)
                y_margin = max(1.0, (max(self.pos_y_data) - min(self.pos_y_data)) * 0.1)
                self.ax1.set_xlim(min(self.pos_x_data) - x_margin, max(self.pos_x_data) + x_margin)
                self.ax1.set_ylim(min(self.pos_y_data) - y_margin, max(self.pos_y_data) + y_margin)
            
            # Update velocity plot
            self.vel_x_line.set_data(list(self.time_data), list(self.vel_x_data))
            self.vel_y_line.set_data(list(self.time_data), list(self.vel_y_data))
            
            if self.time_data:
                self.ax2.set_xlim(max(0, current_time - 30), current_time + 1)
                if self.vel_x_data and self.vel_y_data:
                    vel_max = max(max(self.vel_x_data), max(self.vel_y_data))
                    vel_min = min(min(self.vel_x_data), min(self.vel_y_data))
                    vel_range = max(0.1, vel_max - vel_min)
                    self.ax2.set_ylim(vel_min - vel_range * 0.1, vel_max + vel_range * 0.1)
            
            # Update attitude plot
            self.roll_line.set_data(list(self.time_data), list(self.roll_data))
            self.pitch_line.set_data(list(self.time_data), list(self.pitch_data))
            self.yaw_line.set_data(list(self.time_data), list(self.yaw_data))
            
            if self.time_data:
                self.ax3.set_xlim(max(0, current_time - 30), current_time + 1)
                self.ax3.set_ylim(-180, 180)
            
            # Update GPS plot
            gps_coords = self.tracker.get_path_coordinates()
            if len(gps_coords) > 1:
                lats, lons = zip(*gps_coords)
                self.gps_line.set_data(lons, lats)
                
                # Auto-scale GPS plot
                lat_margin = (max(lats) - min(lats)) * 0.1 or 0.001
                lon_margin = (max(lons) - min(lons)) * 0.1 or 0.001
                self.ax4.set_xlim(min(lons) - lon_margin, max(lons) + lon_margin)
                self.ax4.set_ylim(min(lats) - lat_margin, max(lats) + lat_margin)
            
            self.gps_current.set_data([position.longitude], [position.latitude])
            
            return (self.position_line, self.current_pos, self.vel_x_line, self.vel_y_line,
                   self.roll_line, self.pitch_line, self.yaw_line, self.gps_line, self.gps_current)
            
        except Exception as e:
            print(f"Plot update error: {e}")
            return tuple()
    
    def start_animation(self):
        """Start real-time animation"""
        self.animation = FuncAnimation(
            self.fig, self.update_plot, interval=self.update_interval,
            blit=False, cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function"""
    print("Enhanced IMU Position Tracker v2.0")
    print("=" * 50)
    
    try:
        # Initialize tracker with AprilTag support
        tracker = EnhancedIMUTracker(sample_rate=50.0, enable_apriltag=True)
        
        # Calibrate sensors
        print("\nPress Enter to start calibration (keep IMU stationary)...")
        input()
        tracker.calibrate_sensors(samples=500, duration=10.0)
        
        # Reset position to origin
        tracker.reset_position(0.0, 0.0)
        
        # Start tracking
        tracker.start_tracking()
        
        # Initialize visualizer
        visualizer = PositionVisualizer(tracker, update_interval=50)
        
        print("Starting real-time visualization...")
        print("Close the plot window to stop tracking")
        
        # Start real-time visualization
        try:
            visualizer.start_animation()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Cleanup
        if 'tracker' in locals():
            tracker.stop_tracking()
        print("Tracking stopped")


if __name__ == "__main__":
    main()