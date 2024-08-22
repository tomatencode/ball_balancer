import RPi.GPIO as GPIO
import time

class Servo:
    def __init__(self, pin, offset, max_angle, pwm_min, pwm_max, frequence = 50) -> None:
        self.__offset = offset
        self._angle = offset
        self.__max_angle = max_angle
        self.__pwm_min = pwm_min
        self.__pwm_max = pwm_max

        GPIO.setup(pin, GPIO.OUT)
        self.__pwm = GPIO.PWM(pin, frequence)
        self.__pwm.start(self.angle_to_duty_cycle(offset))

    def __del__(self):
        self.__pwm.stop()
    
    def angle_to_duty_cycle(self, angle):
        if not 0 <= angle <= self.__max_angle:
            raise ValueError(f"angle {angle} not in range 0 - {self.__max_angle}!")

        ratio = (self.__pwm_max - self.__pwm_min)/self.__max_angle #Calcul ratio from angle to percent

        return self.__pwm_min + angle * ratio

    @property
    def angle(self):
        return self._angle
    
    @angle.setter
    def angle(self, value):
        self._angle = min(max(value, -22), 30)
        self.__pwm.ChangeDutyCycle(self.angle_to_duty_cycle(self._angle + self.__offset))
      
if __name__ == "__main__":
    GPIO.setmode(GPIO.BCM)  
    servo1 = Servo(17, 74.3, 180, 2.5, 12.5)
    servo2 = Servo(27, 68.5, 180, 2.5, 12.5)
    servo3 = Servo(22, 90, 180, 2.5, 12.5)

    servo1.angle = -30
    servo2.angle = -30
    servo3.angle = -30

    time.sleep(5)

    del servo1
    GPIO.cleanup()