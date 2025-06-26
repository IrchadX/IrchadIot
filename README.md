# IRCHAD Navigation System – Tag-Based Indoor Guidance

The IRCHAD Navigation System is a Raspberry Pi-powered prototype that provides indoor navigation assistance for visually impaired users. It uses IMU sensors, ultrasonic modules, a Pi camera, and printed AprilTags to estimate the user’s position and guide them in real time within indoor environments.

## Project Structure

Irchadlot/<br>
└── tags/ #examples of tags<br>
├── camera_calibration.py # Calibrate the Pi Camera for AprilTag detection<br>
├── commandes.py # Command interpretation and voice output<br>
├── device_monitoring.py # Hardware status monitoring (IMU, ultrasonic, etc.)<br>
├── navigation.py # Path planning and instruction generation<br>
├── obstacle_detection.py # Uses ultrasonic sensors and YOLOE model for obstacle avoidance<br>
├── position_tracking.py # Position estimation using IMU and AprilTags<br>

## Requirements

- Raspberry Pi 4 with raspbian as OS<br>
- Pi Camera Module v4<br>
- IMU Sensor (MPU6050)<br>
- Ultrasonic Sensor (HC-SR04)<br>
- AprilTags (must be printed and placed in the environment)<br>
- Python 3.8+<br>

## Important Notes

- This system must be run on a Raspberry Pi .<br>
- The first step before using the navigation system is to run `camera_calibration.py` to calibrate the Pi Camera.<br>
- AprilTags must be printed and placed in the environment for accurate position tracking.<br>

## Software Setup

### 1. Enable Interfaces

On Raspberry Pi, run:<br>
sudo raspi-config<br>
Then:<br>

Enable the Camera interface<br>
Enable I2C for the IMU<br>
Reboot the Raspberry Pi<br>

### 2. Set Up Python Environment
sudo apt install python3-venv -y<br>
python3 -m venv irchad-env<br>
source irchad-env/bin/activate<br>
pip install --upgrade pip<br>
### 3. Install Dependencies
sudo apt update && sudo apt upgrade -y<br>
sudo apt install i2c-tools libcamera-apps python3-opencv<br>

pip install numpy scipy matplotlib pillow<br>
pip install opencv-python picamera2<br>
pip install smbus2 mpu6050-raspberrypi adafruit-blinka<br>
pip install RPi.GPIO<br>
pip install pyttsx3<br>
pip install ultralytics dt-apriltags<br>
pip install shapely networkx<br>

## Running the System
### Calibrate the camera for AprilTags:
python3 tags/camera_calibration.py<br>

### Launch the navigation system:
python3 tags/navigation.py<br>
