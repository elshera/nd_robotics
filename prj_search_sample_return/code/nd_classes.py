import numpy as np

#========================================================================================
# Define RoverState() class to retain rover state parameters
#========================================================================================
class RoverState():
    def __init__(self, gtruth):
        # navigation time
        self.start_time         = None        # To record the start time of navigation
        self.total_time         = None        # To record total duration of naviagation

        # map of the area to be explored
        self.ground_truth       = gtruth      # Ground truth worldmap
        
        # current values
        self.img                = None        # Current camera image       
        self.pos                = None        # Current position (x, y)
        self.yaw                = None        # Current yaw angle
        self.pitch              = None        # Current pitch angle
        self.roll               = None        # Current roll angle        
        self.vel                = None        # Current velocity
        self.steer              = 0           # Current steering angle
        self.throttle           = 0           # Current throttle value, acceleration pedal
        self.brake              = 0           # Current brake value
        
        # angles and distances
        self.nav_angles         = None        # Angles of navigable terrain pixels
        self.nav_dists          = None        # Distances of navigable terrain pixels
        self.samples_distance   = None        # Distances of rocks to be collected
        self.samples_angle      = None        # Angles of the rocks to be collected
        
        # controlling parameters
        self.mode               = 'forward'   # Current mode (can be forward or stop)
        self.throttle_set       = 0.2         # Throttle setting when accelerating
        self.brake_set          = 10          # Brake setting when braking
        self.stop_forward       = 50          # Threshold to initiate stopping, The stop_forward and go_forward fields below represent total count of navigable terrain pixels.
        self.go_forward         = 500         # Threshold to go forward again
        self.max_vel            = 10          # Maximum velocity (meters/second)
        

        self.vision_image       = np.zeros((160, 320, 3), dtype=np.float) # Image output from perception step. Update this image to display your intermediate analysis steps on screen in autonomous mode
        self.worldmap           = np.zeros((200, 200, 3), dtype=np.float) # Worldmap
        
        # samples
        self.samples_pos        = None        # To store the actual sample positions
        self.samples_to_find    = 0           # To store the initial count of samples
        self.samples_located    = 0           # To store number of samples located on map
        self.samples_collected  = 0           # To count the number of samples collected
        self.near_sample        = 0           # Will be set to telemetry value data["near_sample"]
        self.samples_available  = 0
        
        self.picking_up         = 0           # Will be set to telemetry value data["picking_up"]
        self.send_pickup        = False       # Set to True to trigger rock pickup


#========================================================================================
# Define data container class to retain rover state parameters
#========================================================================================
# Will read in saved data from csv file and populate this object
# Worldmap is instantiated as 200 x 200 grids corresponding 
# to a 200m x 200m space (same size as the ground truth map: 200 x 200 pixels)
# This encompasses the full range of output position values in x and y from the sim
class Databucket():
    def __init__(self, df, gtruth):
        # replacing some colums with float values
        df['X_Position']  = df['X_Position'].replace('%','',regex=True).astype('float')/100
        df['Y_Position']  = df['Y_Position'].replace('%','',regex=True).astype('float')/100
        df['Yaw']         = df['Yaw'].replace('%','',regex=True).astype('float')/100
        self.images       = df["Path"].tolist()
        self.xpos         = df["X_Position"].values
        self.ypos         = df["Y_Position"].values
        self.yaw          = df["Yaw"].values
        self.speed        = df["Speed"].values
        self.pitch        = df["Pitch"].values
        self.brake        = df["Brake"].values
        self.throttle     = df["Throttle"].values
        self.count        = -1 # This will be a running index
        self.worldmap     = np.zeros((200, 200, 3)).astype(np.float)
        self.ground_truth = gtruth # Ground truth worldmap



