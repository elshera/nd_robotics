import numpy as np
import cv2
from PIL import Image
from io import BytesIO, StringIO
import base64
import time
import math

#========================================================================================
#  Supporting Functions
#========================================================================================
def convert_to_float(string_to_convert):
      if ',' in string_to_convert:
            float_value = np.float(string_to_convert.replace(',','.'))
      else: 
            float_value = np.float(string_to_convert)
      return float_value

#========================================================================================
#  Coordinate manipulation
#========================================================================================
# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel

# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

# define function to translate pixels
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


#========================================================================================
#  Image Filtering
#========================================================================================
# find navigable terrain
def find_navigable(img, rgb_thresh=(160, 160, 160)):
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                 & (img[:,:,1] > rgb_thresh[1]) \
                 & (img[:,:,2] > rgb_thresh[2])
    
    color_select = np.zeros_like(img[:,:,0])    
    color_select[above_thresh] = 1
    return color_select

# find rocks
def find_rocks(img, levels=(110, 110, 50)):
    rockpix =     (img[:,:,0] > levels[0]) \
                & (img[:,:,1] > levels[1]) \
                & (img[:,:,2] < levels[2])

    color_select = np.zeros_like(img[:,:,0])
    color_select[rockpix] = 1
    return color_select

# find obstacles
def find_obstacles(img, rgb_thresh=(140, 140, 140)):
    below_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    
    color_select = np.zeros_like(img[:,:,0])
    color_select[below_thresh] = 1
    return color_select


#========================================================================================
#  Image transformation, warping
#========================================================================================
def perspect_transform(img, src, dst):
    # get the transform function    
    M = cv2.getPerspectiveTransform(src, dst)
    # apply the transform to the input immage by keeping the same size as input image
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0])) 
    # create a mas image where the transform is applied tot a white image
    return warped


#========================================================================================
#  Image composition
#========================================================================================
def create_output_images(Rover):

	# Create a scaled map for plotting and clean up obs/nav pixels a bit
	if np.max(Rover.worldmap[:,:,2]) > 0:
		nav_pix = Rover.worldmap[:,:,2] > 0
		navigable = Rover.worldmap[:,:,2] * (255 / np.mean(Rover.worldmap[nav_pix, 2]))
	else: 
		navigable = Rover.worldmap[:,:,2]

	if np.max(Rover.worldmap[:,:,0]) > 0:
		obs_pix = Rover.worldmap[:,:,0] > 0
		obstacle = Rover.worldmap[:,:,0] * (255 / np.mean(Rover.worldmap[obs_pix, 0]))
	else:
		obstacle = Rover.worldmap[:,:,0]

	likely_nav = navigable >= obstacle
	obstacle[likely_nav] = 0
	plotmap = np.zeros_like(Rover.worldmap)
	plotmap[:, :, 0] = obstacle
	plotmap[:, :, 2] = navigable
	plotmap = plotmap.clip(0, 255)

	# Overlay obstacle and navigable terrain map with ground truth map
	map_add = cv2.addWeighted(plotmap, 1, Rover.ground_truth, 0.5, 0)

	# Check whether any rock detections are present in worldmap
	rock_world_pos = Rover.worldmap[:,:,1].nonzero()
	# If there are, we'll step through the known sample positions
	# to confirm whether detections are real

	samples_located = 0
	if rock_world_pos[0].any():
		rock_size = 2
		for idx in range(len(Rover.samples_pos[0])):
		      test_rock_x = Rover.samples_pos[0][idx]
		      test_rock_y = Rover.samples_pos[1][idx]
		      rock_sample_dists = np.sqrt((test_rock_x - rock_world_pos[1])**2 + (test_rock_y - rock_world_pos[0])**2)                  
		      # If rocks were detected within 3 meters of known sample positions
		      # consider it a success and plot the location of the known
		      # sample on the map
		      if np.min(rock_sample_dists) < 3:
		            samples_located += 1
		            map_add[test_rock_y-rock_size:test_rock_y+rock_size, 
		            test_rock_x-rock_size:test_rock_x+rock_size, :] = 255

	# Calculate some statistics on the map results
	# First get the total number of pixels in the navigable terrain map
	tot_nav_pix = np.float(len((plotmap[:,:,2].nonzero()[0])))
	# Next figure out how many of those correspond to ground truth pixels
	good_nav_pix = np.float(len(((plotmap[:,:,2] > 0) & (Rover.ground_truth[:,:,1] > 0)).nonzero()[0]))
	# Next find how many do not correspond to ground truth pixels
	bad_nav_pix = np.float(len(((plotmap[:,:,2] > 0) & (Rover.ground_truth[:,:,1] == 0)).nonzero()[0]))
	# Grab the total number of map pixels
	tot_map_pix = np.float(len((Rover.ground_truth[:,:,1].nonzero()[0])))
	# Calculate the percentage of ground truth map that has been successfully found
	perc_mapped = round(100*good_nav_pix/tot_map_pix, 1)
	# Calculate the number of good map pixel detections divided by total pixels 
	# found to be navigable terrain
	if tot_nav_pix > 0:
		fidelity = round(100*good_nav_pix/(tot_nav_pix), 1)
	else:
		fidelity = 0
	# Flip the map for plotting so that the y-axis points upward in the display
	map_add = np.flipud(map_add).astype(np.float32)
	# Add some text about map and rock sample detection results
	cv2.putText(map_add,"Time: "+str(np.round(Rover.total_time, 1))+' s', (0, 10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
	cv2.putText(map_add,"Mapped: "+str(perc_mapped)+'%',                  (0, 25), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1) 
	cv2.putText(map_add,"Fidelity: "+str(fidelity)+'%',                   (0, 40), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
	cv2.putText(map_add,"State: "+str(Rover.mode),                        (0, 55), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
	cv2.putText(map_add,"Rocks Located: "+str(samples_located),           (0, 70), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
	cv2.putText(map_add,"Rocks Collected: "+str(Rover.samples_collected), (0, 85), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
	cv2.putText(map_add,"nav_angles: "+str(len(Rover.nav_angles)),        (0, 100), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)	


	# Convert map and vision image to base64 strings for sending to server
	pil_img = Image.fromarray(map_add.astype(np.uint8))
	buff = BytesIO()
	pil_img.save(buff, format="JPEG")
	encoded_string1 = base64.b64encode(buff.getvalue()).decode("utf-8")

	pil_img = Image.fromarray(Rover.vision_image.astype(np.uint8))
	buff = BytesIO()
	pil_img.save(buff, format="JPEG")
	encoded_string2 = base64.b64encode(buff.getvalue()).decode("utf-8")

	return encoded_string1, encoded_string2


#========================================================================================
#  Perception
#========================================================================================
def perception_step(Rover):

    # get the image from the rover
    #=======================================================================
    img  = Rover.img


    # Create source destination arrays for the warping
    #=======================================================================
    dst_size = 5
    bottom_offset = 6

    src = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    dst = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  ])


    # warped image
    #=======================================================================
    warped = perspect_transform(img, src, dst)

    # Apply color threshold to identify navigable terrain/obstacles/rock samples
    #=======================================================================
    nav_map  = find_navigable(warped) 
    obst_map = find_obstacles(warped, rgb_thresh=(140, 140, 140))
    rock_map = find_rocks(warped)

    # Update Rover.vision_image (this will be displayed on left side of screen)
    #=======================================================================
    Rover.vision_image[:,:,0] = obst_map * 255
    Rover.vision_image[:,:,2] = nav_map  * 255


    # coordinate transformation
    #=======================================================================
    world_size = Rover.worldmap.shape[0]
    scale = dst_size * 2
    xpos, ypos = Rover.pos
    yaw = Rover.yaw
    
    # navigable terrain
    xpix_nav, ypix_nav = rover_coords(nav_map)    
    distances_nav, angles_nav = to_polar_coords(xpix_nav, ypix_nav) # Convert to polar coords, calculate distances and angles of navigable terrain
    x_world_nav, y_world_nav = pix_to_world(xpix_nav, ypix_nav, xpos, ypos, yaw, world_size, scale)


    # obstacles
    xpix_obst, ypix_obst = rover_coords(obst_map)
    distances_obst, angles_obst = to_polar_coords(xpix_obst, ypix_obst)
    x_world_obst, y_world_obst = pix_to_world(xpix_obst, ypix_obst, xpos, ypos, yaw, world_size, scale)


    # rocks
    min_dist_rock  = 0.0
    avg_angle_rock = 0.0
    if rock_map.any():
        xpix_rock, ypix_rock = rover_coords(rock_map) 
        distances_rock, angles_rock = to_polar_coords(xpix_rock, ypix_rock)
        x_world_rock, y_world_rock = pix_to_world(xpix_rock, ypix_rock, xpos, ypos, yaw, world_size, scale)


    # Update Rover worldmap (to be displayed on right side of screen)
    #=======================================================================
    if (Rover.pitch < 0.5 or Rover.pitch > 359.5) and (Rover.roll < 0.5 or Rover.roll > 359.5):
	    Rover.worldmap[y_world_nav, x_world_nav, 2]   +=255
	    Rover.worldmap[y_world_obst, y_world_obst, 2] -=255 # this is to improve fidelity
	    
	    Rover.worldmap[y_world_nav, x_world_nav, 0]   -=255 # this is to improve fidelity
	    Rover.worldmap[y_world_obst, x_world_obst, 0] +=255
	    if rock_map.any():
	    	Rover.worldmap[y_world_rock, x_world_rock, 1] =255
    
    
    # Update Rover pixel distances and angles for navigable terrain and rocks
    #=======================================================================
    Rover.nav_dists  = distances_nav
    Rover.nav_angles = angles_nav
    if rock_map.any():
        Rover.samples_available = 1
        Rover.samples_distance = distances_rock
        Rover.samples_angle = angles_rock
        Rover.vision_image[:,:,1] = rock_map * 255
    else:
        Rover.samples_available = 0
        Rover.vision_image[:,:,1] = 0

    # record starting position
    if Rover.start_coords is None:
        Rover.start_coords = (Rover.pos[0], Rover.pos[1])
        print('Rover starting at: ', Rover.start_coords)

    return Rover


#========================================================================================
#  Update the Rover Class from the databucket values
#========================================================================================
def update_rover(Rover, data):
      # Initialize start time and sample positions
      if Rover.start_time == None:
            Rover.start_time = time.time()
            Rover.total_time = 0
            samples_xpos = np.int_([convert_to_float(pos.strip()) for pos in data["samples_x"].split(';')])
            samples_ypos = np.int_([convert_to_float(pos.strip()) for pos in data["samples_y"].split(';')])
            Rover.samples_pos = (samples_xpos, samples_ypos)
            Rover.samples_to_find = np.int(data["sample_count"])
      # Or just update elapsed time
      else:
            tot_time = time.time() - Rover.start_time
            if np.isfinite(tot_time):
                  Rover.total_time = tot_time
      
      # Print out the fields in the telemetry data dictionary
      #print(data.keys())
      # The current speed of the rover in m/s
      Rover.vel = convert_to_float(data["speed"])
      # The current position of the rover
      Rover.pos = [convert_to_float(pos.strip()) for pos in data["position"].split(';')]
      # The current yaw angle of the rover
      Rover.yaw = convert_to_float(data["yaw"])
      # The current yaw angle of the rover
      Rover.pitch = convert_to_float(data["pitch"])
      # The current yaw angle of the rover
      Rover.roll = convert_to_float(data["roll"])
      # The current throttle setting
      Rover.throttle = convert_to_float(data["throttle"])
      # The current steering angle
      Rover.steer = convert_to_float(data["steering_angle"])
      # Near sample flag
      Rover.near_sample = np.int(data["near_sample"])
      # Picking up flag
      Rover.picking_up = np.int(data["picking_up"])
      # Update number of rocks collected
      Rover.samples_collected = Rover.samples_to_find - np.int(data["sample_count"])

      #print('speed =',Rover.vel, 'position =', Rover.pos, 'throttle =', 
      #Rover.throttle, 'steer_angle =', Rover.steer, 'near_sample:', Rover.near_sample, 
      #'picking_up:', data["picking_up"], 'sending pickup:', Rover.send_pickup, 
      #'total time:', Rover.total_time, 'samples remaining:', data["sample_count"], 
      #'samples collected:', Rover.samples_collected)
      
      # Get the current image from the center camera of the rover
      imgString = data["image"]
      image = Image.open(BytesIO(base64.b64decode(imgString)))
      Rover.img = np.asarray(image)

      # Return updated Rover and separate image for optional saving
      return Rover, image


#========================================================================================
#  Cecision - This is where you can build a decision tree for determining throttle, brake and steer
#========================================================================================
def decision_step(Rover):

    # Return Home if we have collected all the samples. I consider returning to the initial position
    # if the rovers enters a circul of radius R (in this case 30) with center the initial position
    if Rover.samples_collected == 6:
        if math.hypot(abs(Rover.pos[0] - Rover.start_coords[0]), abs(Rover.pos[1] - Rover.start_coords[1])) <= 30:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
            Rover.steer = 0
            print('I am Home! Initial Coordinates: (%d, %d). Final coordinates: (%d, %d)'% (Rover.start_coords[0], Rover.start_coords[1], Rover.pos[0], Rover.pos[1]))
            return Rover

    # Rover is stuck so try to get unstuck
    if Rover.mode == 'stuck':
        if time.time() - Rover.stuck_time > (Rover.max_stuck_time + 1):
            Rover.mode = 'forward'
            Rover.stuck_time = time.time()
        else:
            # try to find a good angle
            Rover.throttle = 0
            Rover.brake = 0
            Rover.steer = -15
        return Rover

    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:

    	# Rover has detedted a sample
        #================================================================
        if Rover.getMode() == 'Rock Detected':
            Rover.steer = Rover.getRockAngle()
            if Rover.vel <= 00.5 and Rover.vel >= -0.05:
                #Rover.goToRock()
                Rover.setMode('Go2Rock')
    
        elif Rover.getMode() == 'Go2Rock':
            Rover.goToRock()
            # Check if we are stucked
            if Rover.vel < 0.2 and Rover.throttle != 0:
                # If after throttle, we are not moving then we are stuck
                if time.time() - Rover.stuck_time > Rover.max_stuck_time:
                    # Initiate stuck mode after 5 seconds of not moving
                    Rover.mode = 'stuck'
                    return Rover
            else:
                # Reset stuck time
                Rover.stuck_time = time.time()
            if Rover.near_sample:     
                Rover.stop()
                Rover.setMode('PickingUp')
        
        elif Rover.getMode() == 'PickingUp':
            if Rover.near_sample and (Rover.vel <= 000.5 and Rover.vel >= -0.005) and not Rover.picking_up:      
                Rover.send_pickup = True
            else:
                Rover.steer = 30 if Rover.getNavigationAngle() < 0 else -30
                Rover.setMode('stop')
                 
        elif Rover.samples_available == 1:
            Rover.stop()
            Rover.setMode('Rock Detected')

        # Rover mode is Forward
        #================================================================
        elif Rover.mode == 'forward':
        	# Check if we are stucked
            if Rover.vel < 0.2 and Rover.throttle != 0:
                # If after throttle, we are not moving then we are stuck
                if time.time() - Rover.stuck_time > Rover.max_stuck_time:
                    # Initiate stuck mode after 5 seconds of not moving
                    Rover.mode = 'stuck'
                    return Rover
            else:
                # Reset stuck time
                Rover.stuck_time = time.time()
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15) 
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                # Set mode to "stop" and hit the brakes!
                Rover.throttle = 0
                # Set brake to stored brake value
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = 'stop'

        # Rover mode is Stop
        #================================================================
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    # Check where there is more open terrain, on left or right
                    if Rover.steer == 0:
                        Rover.steer = -15 if np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15) < 0 else 15                                                                                                                        
                # If we're stopped but see sufficient navigable terrain in front then go!
                else:
                	# Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle of mean angle of terrain angles and mean angle of wall angles
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
    
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    return Rover










