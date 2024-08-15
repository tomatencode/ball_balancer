import RPi.GPIO as GPIO
import math
import time
from servo import Servo
from inverse_kinimatics import calc_servo_positions

GPIO.setmode(GPIO.BCM)

servo1 = Servo(17, 74, 180, 2.5, 12.5)
servo2 = Servo(27, 68, 180, 2.5, 12.5)
servo3 = Servo(22, 87, 180, 2.5, 12.5)

angle1, angle2, angle3 = calc_servo_positions(10,-5,120)

servo1.angle = angle1
servo2.angle = angle2
servo3.angle = angle3

time.sleep(5)

servo1.angle = 0
servo2.angle = 0
servo3.angle = 0

time.sleep(0.5)

del servo1
del servo2
del servo3
GPIO.cleanup()