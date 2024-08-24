import math
import numpy as np

def normal_vector_from_projections(theta_x, theta_y):
    """
    Calculate the normal vector given theta_x and theta_y.

    Parameters:
    theta_x (float): The angle between the z-axis and the normal vector's projection onto the XZ plane (in radians).
    theta_y (float): The angle between the z-axis and the normal vector's projection onto the YZ plane (in radians).

    Returns:
    np.array: A 3D normal vector (nx, ny, nz).
    """
    # Calculate the components of the normal vector
    nx = np.sin(theta_x)
    ny = np.sin(theta_y)
    nz = np.cos(theta_x) * np.cos(theta_y)
    
    # Normalize the vector
    normal_vector = np.array([nx, ny, nz])
    normal_vector /= np.linalg.norm(normal_vector)
    
    return normal_vector

def angle_disc_and_rotated_axis(normal_vector, theta_z):
    """
    Calculate the angle between a plane's normal vector and an axis in the XY plane
    rotated by an angle theta_z around the Z-axis.

    Parameters:
    normal_vector (list or array): The normal vector of the plane [n_x, n_y, n_z]
    theta_z (float): The angle by which the axis in the XY plane is rotated around the Z-axis (in radians)

    Returns:
    float: The angle between the plane and the rotated axis in radians
    """
    # Convert normal_vector to a numpy array if it's not already
    n = np.array(normal_vector)
    
    # Calculate the rotated axis vector in the XY plane
    a = np.array([np.cos(theta_z), np.sin(theta_z), 0])
    
    # Calculate the dot product
    dot_product = np.dot(n, a)
    
    # Calculate the magnitudes
    magnitude_n = np.linalg.norm(n)
    magnitude_a = np.linalg.norm(a)
    
    # Calculate the angle in radians
    cos_alpha = dot_product / (magnitude_n * magnitude_a)
    alpha = np.arcsin(cos_alpha)
    
    return alpha

def calc_leg(angle, height):
    # measurements
    lower_arm_len = 70
    upper_arm_len = 109
    disc_radius = 110
    dist_servo_middle = 34.2
    
    # solving triangles
    alpha1 = math.atan(height/dist_servo_middle)
    delta1 = math.atan(dist_servo_middle/height)
    dist_servo_disc = math.sqrt(height**2+dist_servo_middle**2)
    delta = math.radians(90)-delta1 + angle
    arm_len = math.sqrt(dist_servo_disc**2+disc_radius**2-2*dist_servo_disc*disc_radius*math.cos(delta))
    alpha2 = math.asin(disc_radius*math.sin(delta)/arm_len)
    arm_len = min(arm_len,lower_arm_len+upper_arm_len-1)
    alpha3 = math.acos((arm_len**2+lower_arm_len**2-upper_arm_len**2)/(2*arm_len*lower_arm_len))
    alpha = math.radians(180) - (alpha1+alpha2+alpha3)
    
    return alpha

def calc_servo_positions(disc_normal, height):
    disc_angle1 = angle_disc_and_rotated_axis(disc_normal, math.radians(0))
    disc_angle2 = angle_disc_and_rotated_axis(disc_normal, math.radians(120))
    disc_angle3 = angle_disc_and_rotated_axis(disc_normal, math.radians(-120))
    
    angle1 = math.degrees(calc_leg(disc_angle1, height))
    angle2 = math.degrees(calc_leg(disc_angle2, height))
    angle3 = math.degrees(calc_leg(disc_angle3, height))
    
    return angle1, angle2, angle3

if __name__ == "__main__":
    import RPi.GPIO as GPIO
    from servo import Servo
    import time
    
    # GPIO setup
    GPIO.setmode(GPIO.BCM)

    # Initialize servos
    servo1 = Servo(17, 77.3, 180, 2.5, 12.5)
    servo2 = Servo(27, 69.5, 180, 2.5, 12.5)
    servo3 = Servo(22, 90, 180, 2.5, 12.5)
    
    disc_normal = normal_vector_from_projections(math.radians(10), math.radians(0))
    
    angle1, angle2, angle3 = calc_servo_positions(disc_normal, 120)
    servo1.angle = angle1
    servo2.angle = angle2
    servo3.angle = angle3

    # Cleanup
    time.sleep(1)

    del servo1
    del servo2
    del servo3
    GPIO.cleanup()