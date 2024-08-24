import cv2
import numpy as np
import time

class Camera:

    def __init__(self) -> None:
        self.__cap = cv2.VideoCapture(0)
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        before = time.time()
        ret, frame = self.__cap.read()
        after = time.time()
        
        print(f"capture took {int((after-before)*1000)}ms")

        if not ret:
            raise Exception("Could not read camera")

    def preprocess_frame(self, frame, save):
        """ Preprocess the frame to handle lighting and color adjustments. """
        # Adjust gamma to handle lighting conditions
        adjusted_frame = cv2.xphoto.createSimpleWB().balanceWhite(frame)

        # Convert to HSV color space
        hsv = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        
        if save:
            cv2.imwrite("preprocessd_frame.png", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

        return hsv

    def create_mask(self, hsv, save):
        """ Create a mask for detecting the orange ball. """
        lower_orange1 = np.array([0, 150, 136])
        upper_orange1 = np.array([13, 255, 255])
        mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)

        lower_orange2 = np.array([9, 16, 233])
        upper_orange2 = np.array([47, 220, 255])
        mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)

        mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations with the elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        if save:
            cv2.imwrite("mask1.png", mask1)
            cv2.imwrite("mask2.png", mask2)

        return mask

    def find_largest_contour(self, contours, min_contour_area):
        """ Find the largest valid contour based on area and shape metrics. """
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                # Fit an ellipse to the contour
                ellipse = cv2.fitEllipse(contour)
                (x, y), (MA, ma), angle = ellipse

                aspect_ratio = min(MA, ma) / max(MA, ma)
                contour_area = cv2.contourArea(contour)
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                solidity = contour_area / hull_area
                eccentricity = np.sqrt(1 - (MA**2 / ma**2)) if ma > 0 else 0

                # Check if the ellipse meets the conditions
                if 0.65 <= aspect_ratio <= 1.3 and solidity > 0.9 and eccentricity < 0.75:
                    return (x, y), ellipse
        return (None, None), None

    def get_ball_pos_in_frame(self, save=False, use_cam=True, img=None):
        if use_cam:
            ret, frame = self.__cap.read()
            if not ret:
                raise Exception("Could not read camera")
        else:
            frame = img
            
        frame = cv2.resize(frame, (320, 240), interpolation= cv2.INTER_LINEAR)

        hsv = self.preprocess_frame(frame, save)
        mask = self.create_mask(hsv, save)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            min_contour_area = 600  # Adjust based on the expected size of the ball
            (x, y), ellipse = self.find_largest_contour(contours, min_contour_area)

            if x is not None:
                if save:
                    cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                    cv2.imwrite("Detected_Circles.png", frame)
                    cv2.imwrite("mask.png", mask)
                return x, y
        
        if save:
            cv2.imwrite("Detected_Circles.png", frame)
            cv2.imwrite("mask.png", mask)

        return None, None


    def get_ball_pos(self, save=False, use_cam=True, img=None):
        # Get the ball's position in the image frame
        x_in_frame, y_in_frame = self.get_ball_pos_in_frame(save=save, use_cam=use_cam, img=img)
        
        if x_in_frame is not None:
            # Image dimensions
            frame_width = 320  # Adjust if using a different resolution
            frame_height = 240  # Adjust if using a different resolution
            
            frame_center_x = 168
            frame_center_y = 122
            
            # Camera parameters
            fov_x = 140  # Field of view in degrees for the width
            fov_y = fov_x * (frame_height / frame_width)  # Calculate vertical FOV


            # Convert pixel positions to angles
            angle_x = (x_in_frame - frame_center_x) * (fov_x / frame_width)
            angle_y = (y_in_frame - frame_center_y) * (fov_y / frame_height)
            
            # Convert angles from degrees to radians for trigonometric functions
            angle_x_rad = np.radians(angle_x)
            angle_y_rad = np.radians(angle_y)
            
            # Known height of the ball above the camera plane
            h = 140  # in mm
            
            # Compute real-world x and y using trigonometry
            real_x = h * np.tan(angle_x_rad)
            real_y = h * np.tan(angle_y_rad)
            
            real_x_rortated = real_y
            real_y_rortated = -real_x
            
            return real_x_rortated, real_y_rortated
        
        return None, None

    def __del__(self):
        self.__cap.release()

if __name__ == "__main__":
    camera = Camera()

    img = cv2.imread("/home/simon/scr/preprocessd_frame.png")

    x, y =camera.get_ball_pos(save = True, use_cam = True, img = img)
    
    if not x == None:
        print(f"x: {int(x)}, y: {int(y)}, dist: {np.sqrt(x**2 + y**2)}")

    del camera