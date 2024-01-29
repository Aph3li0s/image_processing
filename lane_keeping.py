import numpy as np
import math
import cv2
from collections import defaultdict
import utils.utils_action as action
import ImagePreprocessing
class LaneDetection:
    def __init__(self, opt, debug):
        self.opt = opt["LANE_KEEPING"]
        self.im_height = opt["IMAGE_SHAPE"]["height"]
        self.im_width = opt["IMAGE_SHAPE"]["width"]
        self.debug = debug
    def find_left_right_lane(self, image):
        white_coordinates = np.argwhere(image == 255)[:,::-1]
        white_coordinates = white_coordinates[np.argsort(white_coordinates[::, 0])][::-1] 
        print(white_coordinates)
        
if __name__ == '__main__':
    def display_points(points, image):
        if points is not None:
            image = cv2.circle(image, points, 5, (255, 0, 0), -1)
        return image
    im = cv2.imread(r'test_real/image50.jpg')
    opt = action.load_config_file("main_rc.json")
    direct = LaneDetection(opt, None)
    im_pros = ImagePreprocessing.ImagePreprocessing(opt)
    # lane = im_pros.process_image(im)
    lane = np.array([
                    [0, 0, 255, 0, 0],
                    [255, 0, 255, 255, 255],
                    [0, 255, 0, 0, 0],
                    [0, 255, 0, 255, 0],
                    [0, 0, 0, 0, 0]])
    a = direct.find_left_right_lane(lane)
        