import cv2
import numpy as np
from picamera2 import Picamera2

# Chessboard parameters
chessboard_size = (8, 6)
square_size = 0.025  # in meters

# Termination criteria for subpixel corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare real-world object points
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0],
                       0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points
imgpoints = []  # 2D points

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.configure("preview")
picam2.start()

print("Press 'c' to capture frame for calibration or 'q' to quit")

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    display = frame.copy()

    if ret:
        cv2.drawChessboardCorners(display, chessboard_size, corners, ret)

    cv2.imshow("Calibration", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and ret:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        objpoints.append(objp)
        print("Captured")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()

# Calibrate
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n=== Calibration Results ===")
print("Camera Matrix (mtx):\n", mtx)
print("Distortion Coefficients:\n", dist.ravel())
print("===========================\n")

# Save for later
np.savez("camera_calib.npz", mtx=mtx, dist=dist)
