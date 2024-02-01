import ImagePreprocessing
import IntersectionDetection
import LaneDetection
import LaneKeeping
import time
import numpy as np
import os 
import cv2
import utils.utils_action as action
import math 

def find_coordinate(pointA, pointBB, angle):
    lengthB = math.sqrt(math.pow(pointBB[0] - pointA[0], 2) + math.pow(pointBB[1] - pointA[1], 2))
    Ax, Ay = pointA
    rad = math.radians(angle)
    
    Bx = Ax + lengthB * math.cos(rad)
    By = Ay + lengthB * math.sin(rad)

    return Bx, By

def test_2_img(in_dir):
    opt = action.load_config_file("main_rc.json")
    height = opt["IMAGE_SHAPE"]["height"]
    width = opt["IMAGE_SHAPE"]["width"]
    LaneLine = LaneDetection.LaneDetection(opt)
    LaneKeeper = LaneKeeping.LaneKeeping(opt, True)
    processing_image = ImagePreprocessing.ImagePreprocessing(opt)
    im = cv2.resize(cv2.imread(in_dir), (width, height))    
    lane_det, grayIm = processing_image.process_image(im)            
    
    left_points, right_points, _, _ = LaneLine.find_left_right_lane(lane_det)
    speed, angle, state, lane_data = LaneKeeper.lane_keeping(left_points, right_points)
    left_points = lane_data["left_points"] 
    right_points = lane_data["right_points"] 
    left_point = lane_data["left_point"] 
    right_point = lane_data["right_point"] 
    middle_point = lane_data["middle_point"] 

    new_im = np.copy(im)
    
    if left_point == [0, 0]:
        # By, Bx = find_coordinate(right_point, [width//2, right_point[1]], 23)
        # middle_point = [width//2, height]
        # new_im = cv2.line(new_im, (width//2, height), (int(Bx), int(By)), 255, 2)
        cv2.putText(new_im, 'max right', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    elif right_point == [0, 0]:
        pass 
        # By, Bx = find_coordinate(right_point, [width//2, left_point[1]], 23)
        # middle_point = [width//2, height]
        # new_im = cv2.line(new_im, (width//2, height), (int(Bx), int(By)), 255, 2)
    else:
        new_im = cv2.line(new_im, left_point, right_point, 255, 2)
        new_im = cv2.line(new_im, (width//2, height), middle_point, 255, 2)
    print('speed: ', speed)
    print('angle: ', angle)
    print('state: ', state)
    print('')
    print('left: ', left_point)
    print('right: ', right_point)
    cv2.imshow('a', new_im)
    cv2.waitKey(0)

def test_1_img(in_dir):
    opt = action.load_config_file("main_rc.json")
    height = opt["IMAGE_SHAPE"]["height"]
    width = opt["IMAGE_SHAPE"]["width"]
    check_thresh = opt['INTERCEPT_DETECTION']
    crop_ratio = float(check_thresh['crop_ratio'])
    crop_height_value =  int(height * crop_ratio)
    processing_image = ImagePreprocessing.ImagePreprocessing(opt)
    intersection = IntersectionDetection.IntersectionDetection(opt, debug=True)
    im = cv2.resize(cv2.imread(in_dir), (width, height))    
    im_cut = im[crop_height_value:, :]
    im_cut = processing_image.process_image2(im_cut)
    check_intersection = intersection.detect(im_cut)
    max_lines = check_intersection[1]['max_points']
    for i in max_lines:
        cv2.circle(im, (i[0], i[1] + crop_height_value), 1, (0, 0, 255), -1)
    print(check_intersection[0])
    cv2.imshow('resized image', im)
    cv2.imshow('a', im_cut)
    cv2.waitKey(0)
if __name__ == "__main__":
    test_2_img(r'run_real/frame132.jpg')
