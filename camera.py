import cv2
import numpy as np
import subprocess

class Camera:

    def __init__(self) -> None:
        self.__cap = cv2.VideoCapture(0)
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        subprocess.run(['v4l2-ctl', '--set-ctrl=auto_exposure=1'], check=True)
        subprocess.run(['v4l2-ctl', '--set-ctrl=white_balance_automatic=0'], check=True)
        
        self.__exposure = 150
        self.set_camera_exposure(self.__exposure)

        ret, frame = self.__cap.read()
        
        if not ret:
            raise Exception("Could not read camera")

    def set_camera_exposure(self, exposure_value):
        subprocess.run(['v4l2-ctl', f'--set-ctrl=exposure_time_absolute={exposure_value}'], check=True)
        self.__exposure = exposure_value


    def adjust_exposure(self, frame, ellipse):
        """
        Adjust the camera exposure based on the histogram analysis of the masked area.
        """
        
        mask = np.zeros((240, 320), dtype=np.uint8)
            
        # Unpack ellipse object
        center, axes, rotation_angle = ellipse
        center = (int(center[0]), int(center[1]))
        axes = (int(axes[0] / 2), int(axes[1] / 2))  # Major and minor axes lengths
        rotation_angle = int(rotation_angle)

        # Draw the ellipse on the mask
        cv2.ellipse(mask, center, axes, rotation_angle, 0, 360, 255, -1)

        
        # Apply the mask to the frame to get the relevant region
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        gray_masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        
        # Only consider pixels within the mask (non-zero mask pixels)
        masked_pixels = gray_masked_frame[mask > 0]
        
        # Calculate histogram only for masked pixels
        hist = cv2.calcHist([masked_pixels], [0], None, [256], [0, 256])
        hist = hist / hist.sum()  # Normalize the histogram
        
        peak_value = np.argmax(hist)

        if peak_value < 110:
            self.set_camera_exposure(min(self.__exposure + 10, 2500))
            print(f"Adjusing exposure to {self.__exposure}, peak_bright {peak_value}")
        elif peak_value > 160:
            self.set_camera_exposure(max(self.__exposure - 10, 80))
            print(f"Adjusing exposure to {self.__exposure}, peak_bright {peak_value}")

    def create_mask(self, hsv, save=False):
        """ Create a mask for detecting the orange ball. """
        lower_orange1 = np.array([0, 150, 195])
        upper_orange1 = np.array([35, 255, 255])
        mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)

        lower_orange2 = np.array([9, 16, 233])
        upper_orange2 = np.array([25, 220, 255])
        mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)

        mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations with the elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        if save:
            cv2.imwrite("mask.png", mask)
            cv2.imwrite("mask1.png", mask1)
            cv2.imwrite("mask2.png", mask2)
        
        return mask

    def find_ball_in_contours(self, contours):
        """ Find the largest valid contour based on area and shape metrics. """
        min_contour_area=600
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
                if 0.6 <= aspect_ratio <= 1.4 and solidity > 0.85 and eccentricity < 0.8:
                    return (x, y), ellipse
        return (None, None), None

    def get_ball_pos_in_frame(self, save=False, use_cam=True, adjust_exposure=False, img=None):
        if use_cam:
            ret, frame = self.__cap.read()
            if not ret:
                raise Exception("Could not read camera")
        else:
            frame = img
            
        frame = cv2.resize(frame, (320, 240), interpolation= cv2.INTER_LINEAR)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self.create_mask(hsv, save)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        (x, y), ellipse = self.find_ball_in_contours(contours)
        
        if adjust_exposure and not ellipse == None:
            self.adjust_exposure(frame, ellipse)

        if x is not None:
            if save:
                cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.imwrite("Detected_Circles.png", frame)
            return x, y
        
        if save:
            cv2.imwrite("Detected_Circles.png", frame)

        return None, None

    def get_ball_pos(self, save=False, use_cam=True, adjust_exposure=False, img=None):
        # Get the ball's position in the image frame
        x_in_frame, y_in_frame = self.get_ball_pos_in_frame(save=save, use_cam=use_cam, adjust_exposure=adjust_exposure, img=img)
        
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
            
            return np.array([real_x_rortated, real_y_rortated])
        
        return np.array([np.nan, np.nan])

    def __del__(self):
        self.__cap.release()

if __name__ == "__main__":
    camera = Camera()

    pos = camera.get_ball_pos(save = True)
    
    if not np.isnan(pos).all():
        print(f"x: {int(pos[0])}, y: {int(pos[1])}, dist: {np.sqrt(pos[0]**2 + pos[1]**2)}")

    del camera