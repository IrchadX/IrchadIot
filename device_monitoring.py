#!/usr/bin/env python3

import smbus
import os
import json
import time
import RPi.GPIO as GPIO
from datetime import datetime
import subprocess
import socket
import socketio
import eventlet

# Configuration
STATUS_FILE = "device_status_history.json"
SERVER_PORT = 5000  # Port for WebSocket server
SERVER_IP = "0.0.0.0"  # Listen on all interfaces

# Initialize Socket.IO Server
sio = socketio.Server(cors_allowed_origins='*', async_mode='eventlet')
app = socketio.WSGIApp(sio)

# Connected clients tracker
connected_clients = set()

# Hardware setup
bus = smbus.SMBus(1)
I2C_ADDRESSES = {
    "mpu6050": 0x68,
    "qmc5883l": 0x0D,
    "bmp180": 0x77
}
TRIG = 23
ECHO = 24

# Hardware check functions
def is_device_connected(addr):
    try:
        bus.read_byte(addr)
        return True
    except:
        return False

def is_camera_available():
    try:
        output = subprocess.check_output(["libcamera-hello", "--list-cameras"],
                                       stderr=subprocess.STDOUT, timeout=5).decode("utf-8")
        return "Available cameras" in output or "Registered camera" in output
    except:
        return False

def is_pi_online():
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=5)
        return True
    except:
        return False

def is_ultrasonic_connected():
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG, GPIO.OUT)
        GPIO.setup(ECHO, GPIO.IN)

        GPIO.output(TRIG, False)
        time.sleep(0.05)
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        timeout = time.time() + 0.04
        while GPIO.input(ECHO) == 0:
            if time.time() > timeout:
                GPIO.cleanup()
                return False
        pulse_start = time.time()

        timeout = time.time() + 0.04
        while GPIO.input(ECHO) == 1:
            if time.time() > timeout:
                GPIO.cleanup()
                return False
        pulse_end = time.time()
        GPIO.cleanup()
        return 2 < (pulse_end - pulse_start) * 17150 < 400
    except:
        GPIO.cleanup()
        return False

# Status management
def build_status():
    return {
        "pi_status": "online" if is_pi_online() else "offline",
        "imu": {
            name: "connected" if is_device_connected(addr) else "not_detected"
            for name, addr in I2C_ADDRESSES.items()
        },
        "ultrasonic": "connected" if is_ultrasonic_connected() else "not_detected",
        "camera": "available" if is_camera_available() else "not_detected",
        "timestamp": datetime.utcnow().isoformat()
    }

def create_alert(device_name, status_change):
    now = datetime.now()
    
    # Send to connected mobile clients
    notification = {
        "id": str(int(time.time())),  # Simple timestamp-based ID
        "deviceId": "1",
        "title": device_name,
        "message": status_change,
        "timestamp": now.isoformat(),
        "alertType": device_name,
        "isHandled": False,
        "severity": "modere"  # Default severity
    }
    
    if device_name == "pi_status":
        notification["severity"] = "critique"
    elif device_name in ["camera", "ultrasonic"]:
        notification["severity"] = "modere"
    else:
        notification["severity"] = "mineur"
    
    sio.emit("alert", notification)
    print(f" Sent alert: {notification}")

# Socket.IO event handlers
@sio.event
def connect(sid, environ):
    print(f" Client connected: {sid}")
    connected_clients.add(sid)
    sio.emit('status_update', build_status(), room=sid)

@sio.event
def disconnect(sid):
    print(f" Client disconnected: {sid}")
    if sid in connected_clients:
        connected_clients.remove(sid)

@sio.event
def request_status(sid, data):
    print(f" Status requested by client: {sid}")
    sio.emit('status_update', build_status(), room=sid)

# Main execution
if __name__ == "__main__":
    # Start WebSocket server
    print(f"🚀 Starting HTTP WebSocket server on {SERVER_IP}:{SERVER_PORT}...")
    eventlet.wsgi.server(eventlet.listen((SERVER_IP, SERVER_PORT)), app)