import cv2
import numpy as np
import subprocess

class Camera:
    def __init__(self) -> None:
        """Initialize the camera and set initial properties."""
        self.__cap = cv2.VideoCapture(0)
        
        if not self.__cap.isOpened():
            raise Exception("Could not open camera")

        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.__exposure = 1000
        self.set_camera_exposure(self.__exposure)

        # Ensure initial frame capture
        self.__validate_camera()

    def __validate_camera(self):
        """Validate that the camera is working by capturing a frame."""
        ret, _ = self.__cap.read()
        if not ret:
            self.__cap.release()
            raise Exception("Could not read from camera")

    def set_camera_exposure(self, exposure_value):
        """Set the camera exposure time."""
        try:
            subprocess.run(['v4l2-ctl', f'--set-ctrl=exposure_time_absolute={exposure_value}'], check=True)
            self.__exposure = exposure_value
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to set exposure: {e}")

    def preprocess_frame(self, frame, save=False):
        """Convert frame to HSV color space."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if save:
            # Save preprocessed frame if needed
            pass
        return hsv

    def create_mask(self, hsv, save=False):
        """Create a mask for detecting the orange ball."""
        lower_orange1 = np.array([0, 150, 195])
        upper_orange1 = np.array([35, 255, 255])
        mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)

        lower_orange2 = np.array([9, 16, 233])
        upper_orange2 = np.array([35, 220, 255])
        mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)

        mask = cv2.bitwise_or(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if save:
            cv2.imwrite("mask.png", mask)
            cv2.imwrite("mask1.png", mask1)
            cv2.imwrite("mask2.png", mask2)
        return mask

    def find_ball_in_contours(self, contours):
        """Find the largest valid contour and fit an ellipse."""
        min_contour_area = 600
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                ellipse = cv2.fitEllipse(contour)
                (x, y), (MA, ma), angle = ellipse

                aspect_ratio = min(MA, ma) / max(MA, ma)
                contour_area = cv2.contourArea(contour)
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                solidity = contour_area / hull_area
                eccentricity = np.sqrt(1 - (MA**2 / ma**2)) if ma > 0 else 0

                if 0.6 <= aspect_ratio <= 1.4 and solidity > 0.85 and eccentricity < 0.8:
                    return (x, y), ellipse
        return (None, None), None

    def get_ball_pos_in_frame(self, save=False, use_cam=True, img=None):
        """Get the ball's position in the image frame."""
        if use_cam:
            ret, frame = self.__cap.read()
            if not ret:
                raise Exception("Could not read camera")
        else:
            frame = img

        frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)
        hsv = self.preprocess_frame(frame, save)
        mask = self.create_mask(hsv, save)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (x, y), ellipse = self.find_ball_in_contours(contours)

        if x is not None:
            if save:
                cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.imwrite("Detected_Circles.png", frame)
            return x, y

        if save:
            cv2.imwrite("Detected_Circles.png", frame)
        return None, None

    def get_ball_pos(self, save=False, use_cam=True, img=None):
        """Get the real-world position of the ball."""
        x_in_frame, y_in_frame = self.get_ball_pos_in_frame(save=save, use_cam=use_cam, img=img)
        
        if x_in_frame is not None:
            frame_width = 320
            frame_height = 240
            
            frame_center_x = 168
            frame_center_y = 122
            
            fov_x = 140
            fov_y = fov_x * (frame_height / frame_width)

            angle_x = (x_in_frame - frame_center_x) * (fov_x / frame_width)
            angle_y = (y_in_frame - frame_center_y) * (fov_y / frame_height)

            angle_x_rad = np.radians(angle_x)
            angle_y_rad = np.radians(angle_y)
            
            h = 140

            real_x = h * np.tan(angle_x_rad)
            real_y = h * np.tan(angle_y_rad)
            
            real_x_rortated = real_y
            real_y_rortated = -real_x
            
            return np.array([real_x_rortated, real_y_rortated])
        
        return np.array([np.nan, np.nan])

    def __del__(self):
        """Release the camera resource."""
        if self.__cap.isOpened():
            self.__cap.release()

    def __del__(self):
        self.__cap.release()

if __name__ == "__main__":
    camera = Camera()
    
    img = cv2.imread("/home/simon/scr/preprocessd_frame.png")

    pos = camera.get_ball_pos(save = True, use_cam = True, img = img)
    
    if not np.isnan(pos).all():
        print(f"x: {int(pos[0])}, y: {int(pos[1])}, dist: {np.sqrt(pos[0]**2 + pos[1]**2)}")

    del camera