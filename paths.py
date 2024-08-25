import numpy as np

def middle():
    return np.array([0.0, 0.0])

def line_path(time_running, frequeny=8, line_lenght=60, angle=0):
    
    if time_running%frequeny < frequeny/2:
        
        return np.array([np.cos(np.radians(angle))*line_lenght, np.sin(np.radians(angle))*line_lenght])
    else:
        
        return np.array(-[np.cos(np.radians(angle))*line_lenght, -np.sin(np.radians(angle))*line_lenght])

def random_path(time_running, last_point, dt, frequency=8, radius=60):
    
    if time_running%frequency > 0 and time_running%frequency-dt < 0:
        target_pos = np.random.rand((1,2))*radius
        
        while np.linalg.norm(target_pos) > radius:
            target_pos = np.random.rand((1,2))*radius
        
        return target_pos
    else:
        return last_point