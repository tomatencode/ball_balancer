import math

def get_slope_in_direction_on_plate(x_rot, y_rot, angle):
    
    if math.asin(math.radians(x_rot)) == 0:
        if math.asin(math.radians(y_rot)) > 0:
            phase = 3 * math.pi / 2
            amplitude = math.asin(math.radians(y_rot))
        else:
            phase = math.pi / 2
            amplitude = -math.asin(math.radians(y_rot))
    else:
        phase = math.atan2(-math.asin(math.radians(y_rot)), math.asin(math.radians(x_rot)))
        amplitude = math.asin(math.radians(x_rot)) / math.cos(phase)
    
    return math.asin(math.cos(phase+math.radians(angle))*amplitude)

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
    alpha3 = math.acos((arm_len**2+lower_arm_len**2-upper_arm_len**2)/(2*arm_len*lower_arm_len))
    alpha = math.radians(180) - (alpha1+alpha2+alpha3)
    
    return alpha

def calc_servo_positions(x_rot, y_rot, height):
    disc_angle1 = get_slope_in_direction_on_plate(x_rot, y_rot, 0)
    disc_angle2 = get_slope_in_direction_on_plate(x_rot, y_rot, 120)
    disc_angle3 = get_slope_in_direction_on_plate(x_rot, y_rot, -120)
    
    angle1 = math.degrees(calc_leg(disc_angle1, height))
    angle2 = math.degrees(calc_leg(disc_angle2, height))
    angle3 = math.degrees(calc_leg(disc_angle3, height))
    
    return angle1, angle2, angle3