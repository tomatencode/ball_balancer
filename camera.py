import cv2 
import numpy as np 

# Open the webcam (index 0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Capture a single frame
ret, frame = cap.read()

# Check if the frame was captured successfully
if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

img = frame

# Convert to grayscale. 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur using a smaller kernel
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 2)

# Apply Hough transform on the blurred image
detected_circles = cv2.HoughCircles(
    gray_blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=50,
    param1=100,
    param2=60,
    minRadius=30,
    maxRadius=250
)

# Draw circles that are detected. 
if detected_circles is not None: 

	# Convert the circle parameters a, b and r to integers. 
	detected_circles = np.uint16(np.around(detected_circles)) 

	for pt in detected_circles[0, :]: 
		a, b, r = pt[0], pt[1], pt[2] 

		# Draw the circumference of the circle. 
		cv2.circle(img, (a, b), r, (0, 255, 0), 2) 

		# Draw a small circle (of radius 1) to show the center. 
		cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
  
cv2.imwrite("circles.jpg", img)
cap.release()
