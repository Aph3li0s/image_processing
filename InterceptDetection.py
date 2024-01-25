import numpy as np
import cv2
from collections import defaultdict

class InterceptDetection:
    def __init__(self, opt, debug=False):
        self.opt = opt["INTERCEPT_DETECTION"]
        self.debug = debug


    def find_maximum_connected_line(self, sybinary):
        white_pixel_idx = np.argwhere(sybinary == 255)[:, ::-1]    
        white_pixel_idx  = white_pixel_idx[np.argsort(white_pixel_idx[::, 0])]
        
        white_map = defaultdict(list)                                                                                 
        for point in white_pixel_idx:                                                     
            white_map[point[0]].append(point)     

        new_points = []
        for x_idx in white_map:     
            points = white_map[x_idx]    
            if len(points) >= self.opt["minimum_points"]:
                new_point = np.mean(points, axis = 0, dtype=np.int32)
                new_points.append(new_point)

        max_len = 0
        max_points = []
        
        if len(new_points) > 0:
            new_points = np.array(new_points)
            current_x = new_points[0, 0]
            max_len = 0
            current_len = 0
            max_points = [new_points[0]]
            current_points = [new_points[0]]

            for point in new_points[1:]:
                if point[0] <= self.opt["tolerance"] + current_x:
                    current_points.append(point)
                    current_len += 1
                
                else:
                    if current_len >= max_len:
                        max_len = current_len
                        max_points = current_points
                    current_len = 1
                    current_points = [point]

                current_x = point[0]

            if current_len > max_len:
                max_len = current_len
                max_points = current_points

        gap = float("inf")
            
        if len(max_points) > 0:
            max_points = np.array(max_points)
            gap = np.max(max_points[:, 1]) - np.min(max_points[:, 1])
        return max_len, gap, max_points

    def detect(self, sybinary):
        image_height = sybinary.shape[0]
        sybinary = sybinary[int(image_height * self.opt["crop_ratio"]):, :]
        
        max_len, gap, max_points = self.find_maximum_connected_line(sybinary)
        if self.debug:
            debug_data = {"image_size" : [int(sybinary.shape[0]), int(sybinary.shape[1])],
                        "max_points" : [[int(point[0]), int(point[1])] for point in max_points]}

            return [max_len, gap], debug_data

        return [max_len, gap], None
