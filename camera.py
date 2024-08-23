import cv2
import numpy as np

class Camera:

    def __init__(self) -> None:
        self.__cap = cv2.VideoCapture(0)
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        ret, frame = self.__cap.read()

        if not ret:
            raise Exception("Could not read camera")

    def preprocess_frame(self, frame):
        """ Preprocess the frame to handle lighting and color adjustments. """
        # Adjust gamma to handle lighting conditions
        adjusted_frame = cv2.xphoto.createSimpleWB().balanceWhite(frame)

        # Convert to HSV color space
        hsv = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        return hsv

    def create_mask(self, hsv):
        """ Create a mask for detecting the orange ball. """
        lower_orange1 = np.array([0, 110, 136])
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

        hsv = self.preprocess_frame(frame)
        mask = self.create_mask(hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            min_contour_area = 600  # Adjust based on the expected size of the ball
            (x, y), ellipse = self.find_largest_contour(contours, min_contour_area)

            if x is not None:
                if save:
                    contour_image = frame.copy()
                    cv2.drawContours(contour_image, contours[0], -1, (0, 255, 0), 2)
                    cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                    cv2.imwrite("Detected_Circles.png", frame)
                    cv2.imwrite("mask.png", mask)
                return x, y

        return None, None

    def get_ball_pos(self, save=False, use_cam=True, img=None):
        x_in_frame, y_in_frame = self.get_ball_pos_in_frame(save=save, use_cam=use_cam, img=img)
        if x_in_frame is not None:
            y = x_in_frame - 164
            x = y_in_frame - 120
            y = -y
            return x, y
        return None, None

    def __del__(self):
        self.__cap.release()

if __name__ == "__main__":
    camera = Camera()

    img = cv2.imread("/home/simon/src/ball_balancer-1/adjusted_frame.png")

    camera.get_ball_pos(save = True, use_cam = False, img = img)

    del camera