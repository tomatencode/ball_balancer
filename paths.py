import numpy as np
import math

def middle():
    return np.array([0.0, 0.0])

def line_path(time_running, frequeny=8.0, line_lenght=40.0, angle=90.0):
    
    if time_running%frequeny < frequeny/2:
        
        return np.array([math.cos(math.radians(angle))*line_lenght, math.sin(math.radians(angle))*line_lenght])
    else:
        
        return np.array([-math.cos(math.radians(angle))*line_lenght, -math.sin(math.radians(angle))*line_lenght])

def circle_path(time_running, radius=50.0, frequency=10.0):
    
    phi = (time_running%frequency)/frequency*360
    
    return np.array([math.cos(math.radians(phi))*radius, math.sin(math.radians(phi))*radius])

def random_path(time_running, last_point, dt, frequency=6, radius=30.0):
    
    if time_running%frequency > 0 and time_running%frequency-dt < 0:
        target_pos = (np.random.rand(2,)-0.5)*radius*2
        
        while np.linalg.norm(target_pos) > radius:
            target_pos = (np.random.rand(2,)-0.5)*radius
        
        return target_pos
    else:
        return last_point