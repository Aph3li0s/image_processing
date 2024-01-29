import time
import numpy as np
import math
import cv2
from utils.utils_function import calculate_speed
from collections import defaultdict
class LaneKeeping:
    def __init__(self, opt, debug):
        self.opt = opt["LANE_KEEPING"]
        self.im_height = opt["IMAGE_SHAPE"]["height"]
        self.im_width = opt["IMAGE_SHAPE"]["width"]
        self.debug = debug
        self.prev_angle = 0
        self.prev_state = 0
        
    def fit_2d(self, fit, pointX):                                                          # F(x) = ax^2 + bx + c, fit = [a, b, c]
        pointY =  int(fit[2] + fit[1] * pow(pointX, 1) + fit[0] * pow(pointX, 2))
        return pointY
    

    def lane_keeping(self, left_points, right_points):

        # Init dummy variables
        left_point_x, left_point_y = 0, 0           
        right_point_x, right_point_y = 0, self.im_width - 1
        point_y = 0
        
        if len(left_points) != 0:                                                           # If there is a left lane
            left_points = np.array(left_points)

            # If the height of lane is enough
            if abs(np.max(left_points[:, 1]) - np.min(left_points[:, 1])) > self.opt["min_length"]:             
                left_points = left_points                                                   # Keep the left lane
            else:
                left_points = []                                                            # Else remove short left lane

        if len(right_points) != 0:                                                          # If there is a right lane
            right_points = np.array(right_points)                       

            # If the height of lane is enough
            if abs(np.max(right_points[:, 1]) - np.min(right_points[:, 1])) > self.opt["min_length"]:           
                right_points = right_points                                                 # Keep the right lane
            else:
                right_points = []                                                           # Else remove short right lane

        if len(left_points) == 0 and len(right_points) == 0:                                # If there is no lane
            state = self.prev_state                                                         # Remain the same angle and the same state
            angle = self.prev_angle
            middle_point_x = self.im_width //2 

        elif len(left_points) == 0 and len(right_points) != 0:                              # If there is only right lane
            state = -1                                                                      # Turn max angle to the left                   
            angle = -23                                                                     # Assign state = -1 means missing left lane   
            middle_point_x = 0
        
        elif len(left_points) != 0 and len(right_points) == 0:                              # If there is only left lane
            state = 1                                                                       # Turn max angle to the right   
            angle = 23                                                                      # Assign state = 1 means missing right lane   
            middle_point_x = self.im_width-1
        
        else:
            state = 0                                                                       # If there are 2 lanes, Assign state = 0 means fully visible
            left_line = np.polyfit(left_points[:,1], left_points[:, 0], 2)                  # Find the 2d polynomial function for all left point
            
            # Get 1st quartile point between max height pixel and min height pixel in left lane
            left_point_y = int(abs(np.max(left_points[:, 1]) - np.min(left_points[:, 1])) * self.opt["middle_point_ratio"] \
                                                                                                + np.min(left_points[:, 1]))

            right_line = np.polyfit(right_points[:,1], right_points[:, 0], 2)               # Find the 2d polynomial function for all right point
            # Get 1st quartile point between max height pixel and min height pixel in left lane
            right_point_y = int(abs(np.max(right_points[:, 1]) - np.min(right_points[:, 1])) * self.opt["middle_point_ratio"] \
                                                                                                + np.min(right_points[:, 1]))

            # Get the middle height between left_point_y and right_point_y
            point_y = int((left_point_y  + right_point_y)/2)

            # Find the x value of 2 1st quartile point from left, right, and the middle point
            left_point_x = self.fit_2d(left_line, left_point_y)
            right_point_x = self.fit_2d(right_line, right_point_y)
            middle_point_x = int((left_point_x + right_point_x)/2)

            dx = self.im_width//2 - middle_point_x                                                      # Calculate error angle 
            if dx != 0:
                dy = self.im_height - point_y
                angle =  math.atan(dy/dx) * 180 / math.pi
                if angle >= 0:
                    angle = - (90 - angle)
                else:
                    angle = 90 +  angle
            else:
                angle = 0

            angle = angle * self.opt["angle_scale_ratio"]                                   # Scale angle to the small value to maintain the stability    
            
        angle = np.clip(angle, -23, 23)     
        if np.abs(self.prev_angle - angle) > 23:
            angle = self.prev_angle
        else:
            self.prev_angle = angle

        speed = calculate_speed(angle, max_speed = 100)                                     # Calculate speed using gaussian function

        if self.debug:
            debug_data = {"angle": int(angle),
                        "image_size" : [int(self.im_height), int(self.im_width)],
                        "left_points" : [list([int(point[0]), int(point[1])]) for point in left_points],
                        "right_points" : [list([int(point[0]), int(point[1])]) for point in right_points],
                        "left_point" : [left_point_x, left_point_y],
                        "right_point" : [right_point_x, right_point_y],
                        "middle_point" : [middle_point_x, point_y]}
        
            return speed, angle, state, debug_data

        lane_data = { "image_size" : [int(self.im_height), int(self.im_width)],
                        "left_points" : left_points,
                        "right_points" : right_points,
                        "left_point" : [left_point_x, left_point_y],
                        "right_point" : [right_point_x, right_point_y]}
        
        
        return speed, angle, state, lane_data
        
        



