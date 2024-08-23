import cv2
import numpy as np

class Camera:

    def __init__(self) -> None:
        self.__cap = cv2.VideoCapture(0)
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        ret, frame = self.__cap.read()

        if not ret:
            raise Exception("Culd not read camera")

    def get_ball_pos_in_frame(self, save=False):
        ret, frame = self.__cap.read()

        if not ret:
            raise Exception("Culd not read camera")
            
        # Adjust gamma to handle lighting conditions
        adjusted_frame = cv2.xphoto.createSimpleWB().balanceWhite(frame)

        # Convert to HSV color space
        hsv = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        # Define the orange color range in HSV
        lower_orange1 = np.array([0, 120, 115])
        upper_orange1 = np.array([4, 255, 255])
        mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)

        lower_orange2 = np.array([0, 87, 175])
        upper_orange2 = np.array([15, 255, 210])
        mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)

        mask = cv2.bitwise_or(mask1, mask2)

        # Apply Gaussian blur to reduce noise
        blurred_mask = cv2.GaussianBlur(mask, (9, 9), 2)

        # Find contours in the mask
        contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        if contours:
            # Find the largest contour (assuming it's the ball)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            min_contour_area = 1000  # Adjust based on the expected size of the ball
            
            # Iterate through the contours
            for contour in contours:
                if cv2.contourArea(contour) > min_contour_area:
                    # Fit an ellipse to the largest contour to account for lens distortion
                    ellipse = cv2.fitEllipse(contour)
                    
                    (x, y), (MA, ma), angle = ellipse
                    
                    # Calculate the aspect ratio (MA/ma for the ellipse)
                    aspect_ratio = min(MA, ma) / max(MA, ma)
                    
                    # Calculate the area and convex hull of the contour
                    contour_area = cv2.contourArea(contour)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    
                    # Calculate solidity (contour_area / hull_area)
                    solidity = contour_area / hull_area
                    
                    # Calculate the eccentricity of the ellipse
                    eccentricity = np.sqrt(1 - (MA**2 / ma**2)) if ma > 0 else 0
                    
                    # Conditions to accept the ellipse based on aspect ratio, solidity, and eccentricity
                    if 0.7 <= aspect_ratio <= 1.3 and solidity > 0.9 and eccentricity < 0.7:
                    
                        if save:
                            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                            cv2.imwrite("Detected_Circles.png", frame)
                            cv2.imwrite("adjusted_frame.png", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
                            cv2.imwrite("mask.png", mask)
                            cv2.imwrite("mask1.png", mask1)
                            cv2.imwrite("mask2.png", mask2)
                        
                        return x, -y
                
                else:
                    break
            
            if save:
                cv2.imwrite("Detected_Circles.png", frame)
                cv2.imwrite("adjusted_frame.png", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
                cv2.imwrite("mask.png", mask)
                cv2.imwrite("mask1.png", mask1)
                cv2.imwrite("mask2.png", mask2)
            
            return None, None
        else:
            if save:
                cv2.imwrite("Detected_Circles.png", frame)
                
            return None, None
    
    def get_ball_pos(self,save=False):
        x_in_frame, y_in_frame = self.get_ball_pos_in_frame(save=save)
        if not x_in_frame == None:
            # center of plate
            y = x_in_frame-164
            x = y_in_frame-120
            
            return x, y
        else:
            return None, None
        

    def __del__(self):
        self.__cap.release()

if __name__ == "__main__":
    camera = Camera()

    camera.get_ball_pos(save=True)

    del camera