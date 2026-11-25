# shall bring the dataset to a common ground 

# we convert screen coordinates to deltas since they depend from screen
# size and we dont want model to focus on those changes in motion 
# rather than different screen behaviour.

# so we compute dx, dy and dt from consecuive events:
# dx_i = x_i - x_{i-1} horizontal step in pixels 

# we can also compute:
# step speed: sqrt{ dx^2 + dy^2 } / dt
# speed: d/t 
# acceleration: derivative of speed
# Direction/angle change:  np.arctan2(dy, dx) then compute angle change between 2 timesteps 
# jerk: first derivative of acceleration 


def read_samples():
    pass

def process_sample():
    pass

def normalize_features():
    pass