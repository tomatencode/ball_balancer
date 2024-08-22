import RPi.GPIO as GPIO
from servo import Servo
import threading as th
import time
from inverse_kinimatics import calc_servo_positions
from camera import Camera

# GPIO setup
GPIO.setmode(GPIO.BCM)

# Initialize servos
servo1 = Servo(17, 77.3, 180, 2.5, 12.5)
servo2 = Servo(27, 69.5, 180, 2.5, 12.5)
servo3 = Servo(22, 90, 180, 2.5, 12.5)
camera = Camera()

# PID parameters
p = 0.11
i = 0.04
d = 0.07
max_integral = 100  # Limit for integral term
ball_on_plate = False

# State variables
pos_history = []
history_len = 2
integral_x = 0
integral_y = 0
last_time = time.time()

# Flags
running = True

# Key capture thread to stop the program
def key_capture_thread():
    global running
    input()
    running = False

# saves the balls position
def record_hisory(x, y):
    pos_history.insert(0, [x, y, time.time()])
    while len(pos_history) > history_len:
        pos_history.pop(history_len)

# Get ball velocity
def get_ball_vel():
    if len(pos_history) == history_len:
        x_diff = pos_history[0][0] - pos_history[history_len-1][0]
        y_diff = pos_history[0][1] - pos_history[history_len-1][1]
        dt = time.time() - pos_history[history_len-1][2]
        v_x = x_diff/dt
        v_y = y_diff/dt
    else:
        v_x, v_y = 0, 0
    return v_x, v_y

th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
while running:
    
    x, y = camera.get_ball_pos()
    
    if x is not None and y is not None:
        if not ball_on_plate:
            ball_on_plate = True
            print("ball on plate")
        
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        record_hisory(x, y)

        error_x = x  # target position is 0,0
        error_y = y

        integral_x += error_x * dt
        integral_x = min(max(integral_x, -max_integral), max_integral)
        
        integral_y += error_y * dt
        integral_y = min(max(integral_y, -max_integral), max_integral)

        v_x, v_y = get_ball_vel()
        
        slope_x = p*error_x + i*integral_x + d*v_x
        slope_y = p*error_y + i*integral_y + d*v_y
        
        slope_x = min(max(slope_x, -20), 20)
        slope_y = min(max(slope_y, -20), 20)
        
        angle1, angle2, angle3 = calc_servo_positions(slope_x, slope_y, 110)
        servo1.angle = angle1
        servo2.angle = angle2
        servo3.angle = angle3
    else:
        if ball_on_plate:
            ball_on_plate = False
            print("ball fell off :(")
        servo1.angle = 0
        servo2.angle = 0
        servo3.angle = 0

# Cleanup
servo1.angle = 0
servo2.angle = 0
servo3.angle = 0

time.sleep(1)

del camera
del servo1
del servo2
del servo3
GPIO.cleanup()