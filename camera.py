import cv2
import numpy as np
from datetime import datetime

mask_scaling_factor = 3

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

ret, frame = cap.read()
ret, frame = cap.read()

if not ret:
    exit()

start = datetime.now()

# scale down the mask to make the cirvle detection faster
frame_widht, frame_height, _ = frame.shape
adjusted_frame = cv2.resize(frame, (int(frame_height/mask_scaling_factor), int(frame_widht/mask_scaling_factor)))

# Adjust gamma to handle lighting conditions
adjusted_frame = cv2.xphoto.createSimpleWB().balanceWhite(adjusted_frame)

# Convert to HSV color space
hsv = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)
hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

# Define the color range for detection
target_color_hsv = np.array([55, 170, 170])  # ball color
lower_color, upper_color = adjust_hsv_range(hsv, target_color_hsv)
mask = cv2.inRange(hsv, lower_color, upper_color)


# Blur the mask to reduce noise
mask = cv2.GaussianBlur(mask, (max(3, int(9/mask_scaling_factor)), max(3, int(9/mask_scaling_factor))), 2)


# Detect edges using Canny
edges = cv2.Canny(mask, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Approximate the contour to a circle
    center, radius = cv2.minEnclosingCircle(contour)

    if radius > 5:  # Filter out small circles
        x, y = tuple(map(int, center))
        radius = int(radius*mask_scaling_factor)

        # Draw the circle
        cv2.circle(frame, (x*mask_scaling_factor,y*mask_scaling_factor), radius, (0, 255, 0), 2)
        cv2.circle(frame, (x*mask_scaling_factor,y*mask_scaling_factor), 2, (0, 0, 255), 3)
        
duration = datetime.now()-start

# Save the processed images
cv2.imwrite("mask.png", mask)
cv2.imwrite("adjusted_frame.png", adjusted_frame)
cv2.imwrite("Detected_Circles.png", frame)
print(f"tps: {1000000/duration.microseconds}")

cap.release()