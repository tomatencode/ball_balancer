import cv2
import numpy as np
from datetime import datetime

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_hsv_range(hsv, target_color_hsv):
    # Calculate the mean value of the HSV image
    mean_hue = np.mean(hsv[:, :, 0])
    mean_saturation = np.mean(hsv[:, :, 1])
    mean_value = np.mean(hsv[:, :, 2])

    # Adjust the target color based on the current lighting
    lower_bound = np.array([
        max(0, target_color_hsv[0] - 20),
        max(0, target_color_hsv[1] - 100),
        max(0, target_color_hsv[2] - 100)
    ])
    upper_bound = np.array([
        min(190, target_color_hsv[0] + 20),
        min(255, target_color_hsv[1] + 100),
        min(255, target_color_hsv[2] + 100)
    ])
    return lower_bound, upper_bound

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

ret, frame = cap.read()

start = datetime.now()
ret, frame = cap.read()

if not ret:
    exit()

# Adjust gamma to handle lighting conditions
adjusted_frame = cv2.xphoto.createSimpleWB().balanceWhite(frame)

# Convert to HSV color space
hsv = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)
hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

# Define the color range for detection
target_color_hsv = np.array([55, 170, 170])  # ball color
lower_color, upper_color = adjust_hsv_range(hsv, target_color_hsv)
mask = cv2.inRange(hsv, lower_color, upper_color)


# Apply Gaussian blur
blurred_mask = cv2.GaussianBlur(mask, (9, 9), 2)

# Perform Canny edge detection
edges = cv2.Canny(blurred_mask, 50, 150)

# Detect circles using Hough Transform
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=250)
        
duration = datetime.now()-start

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 1)
        cv2.circle(frame, (i[0], i[1]), 0, (0, 0, 255), 1)

# Save the processed images
cv2.imwrite("mask.png", blurred_mask)
cv2.imwrite("adjusted_frame.png", adjusted_frame)
cv2.imwrite("Detected_Circles.png", frame)
print(f"tps: {1000000/duration.microseconds}")

cap.release()