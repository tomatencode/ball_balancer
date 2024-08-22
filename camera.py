import cv2
import numpy as np

class Camera:

    def __init__(self) -> None:
        self.__cap = cv2.VideoCapture(0)
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        ret, frame = self.__cap.read()

        if not ret:
            exit()

    def get_ball_pos_in_frame(self, save=False):
        ret, frame = self.__cap.read()

        if not ret:
            exit()
            
        # Adjust gamma to handle lighting conditions
        adjusted_frame = cv2.xphoto.createSimpleWB().balanceWhite(frame)

        # Convert to HSV color space
        hsv = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        # Define the orange color range in HSV
        lower_orange1 = np.array([0, 193, 136])
        upper_orange1 = np.array([13, 255, 255])
        mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)

        lower_orange2 = np.array([9, 16, 233])
        upper_orange2 = np.array([47, 220, 255])
        mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)

        mask = cv2.bitwise_or(mask1, mask2)

        # Apply Gaussian blur to reduce noise
        blurred_mask = cv2.GaussianBlur(mask, (9, 9), 2)

        # Find contours in the mask
        contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        if contours:
            # Find the largest contour (assuming it's the ball)
            largest_contour = max(contours, key=cv2.contourArea)
            
            min_contour_area = 1000  # Adjust based on the expected size of the ball
            if cv2.contourArea(largest_contour) > min_contour_area:
                # Fit an ellipse to the largest contour to account for lens distortion
                ellipse = cv2.fitEllipse(largest_contour)
                
                center = (int(ellipse[0][0]), int(ellipse[0][1]))
                
                if save:
                    cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
                    cv2.circle(frame, center, 3, (0, 0, 255), -1)
                    cv2.imwrite("Detected_Circles.png", frame)
                    cv2.imwrite("mask.png", mask)
                    cv2.imwrite("mask1.png", mask1)
                    cv2.imwrite("mask2.png", mask2)
                    
                return center
                
            if save:
                cv2.imwrite("Detected_Circles.png", frame)
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
            
            # because camera is rortated
            # and the order of servos
            # idk it works
            y = -y
            
            return x, y
        else:
            return None, None
        

    def __del__(self):
        self.__cap.release()

if __name__ == "__main__":
    camera = Camera()

    camera.get_ball_pos(save=True)

    del camera