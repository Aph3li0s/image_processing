import numpy as np
import cv2
from collections import defaultdict

from utils.utils_function import display_lines, get_point, display_points, connect_lines_y_axis, connect_lines_x_axis

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
        max_len, gap, max_points = self.find_maximum_connected_line(sybinary)
        if self.debug:
            debug_data = {"image_size" : [int(sybinary.shape[0]), int(sybinary.shape[1])],
                        "max_points" : [[int(point[0]), int(point[1])] for point in max_points]}

            return [max_len, gap], debug_data

        return [max_len, gap], None
    
if __name__ == "__main__":
    test_im = r'test_real/image120.jpg'
    import os
    import ImagePreprocessing
    import cv2
    import utils.utils_action as action
    import utils.utils_function as func
    opt = action.load_config_file("main_rc.json")
    im_pros = ImagePreprocessing.ImagePreprocessing(opt)
    intercept = InterceptDetection(opt, debug=True)
    im = cv2.imread(test_im)
    im_cut = im[int(480 * 0.35):, :]
    height = int(im.shape[0] * float(opt['INTERCEPT_DETECTION']['crop_ratio']))
    result= im_pros.process_image2(im_cut)
    check_intercept, debug = intercept.detect(result)
    # print('detected' if check_intercept[0][0] > 300 else 'not detected')
    a = debug['max_points']
    for i in a:
        cv2.circle(im, (i[0], i[1] + height), 2, (0, 0, 255), -1)
    print(check_intercept)
    # dis_lines = func.connect_lines_y_axis(a)
    cv2.imshow('test', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
