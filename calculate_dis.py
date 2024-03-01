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

def display_points(points, image, color):
    if color == 0:
        if points is not None:
            for point in points:
                image = cv2.circle(image, point, 1, (255, 0, 0), -1)    #red
        return image
    if color == 1:
        if points is not None:
            for point in points:
                image = cv2.circle(image, point, 1, (0, 0, 255), -1)    #red
        return image

def find_coordinate(pointA, pointBB, angle):
    lengthB = math.sqrt(math.pow(pointBB[0] - pointA[0], 2) + math.pow(pointBB[1] - pointA[1], 2))
    Ax, Ay = pointA
    rad = math.radians(angle)
    
    Bx = Ax + lengthB * math.cos(rad)
    By = Ay + lengthB * math.sin(rad)

    return Bx, By
    
def test_lane_img(in_dir):
    opt = action.load_config_file("main_rc.json")
    check_thresh = opt['INTERSECT_DETECTION']
    crop_ratio = float(check_thresh['crop_ratio'])
    height = opt["IMAGE_SHAPE"]["height"]
    width = opt["IMAGE_SHAPE"]["width"]
    LaneLine = LaneDetection.LaneDetection(opt)
    LaneKeeper = LaneKeeping.LaneKeeping(opt, True)
    processing_image = ImagePreprocessing.ImagePreprocessing(opt)
    # Resize down to (320, 240)
    im = cv2.resize(cv2.imread(in_dir), (height, width)) 
    crop_height_value =  int(height * crop_ratio)
    im_cut = im[crop_height_value:, :]
    lane_det, grayIm = processing_image.process_image(im)         

    left_points, right_points, _, _ = LaneLine.find_left_right_lane(lane_det)

    speed, angle, state, lane_data = LaneKeeper.lane_keeping(left_points, right_points)
    left_points = lane_data["left_points"] 
    right_points = lane_data["right_points"] 
    left_point = lane_data["left_point"] 
    right_point = lane_data["right_point"] 
    middle_point = lane_data["middle_point"] 
    
    new_im = np.copy(im)
    new_im = display_points(left_points, im, 0)
    new_im = display_points(right_points, im, 1)
    # print('speed: ', speed)
    # print('angle: ', angle)
    # print('state: ', state)q
    # print('')
    
    cv2.imshow('raw_im', processing_image.region_of_interest(im))
    cv2.imshow('lane_det', lane_det)
    cv2.imshow('intersect', im_cut)
    cv2.waitKey(0)
    
def test_1_img(in_dir):
    opt = action.load_config_file("main_rc.json")
    height = opt["IMAGE_SHAPE"]["height"]
    width = opt["IMAGE_SHAPE"]["width"]
    check_thresh = opt['INTERSECT_DETECTION']
    crop_ratio = float(check_thresh['crop_ratio'])
    crop_height_value =  int(height * crop_ratio)
    processing_image = ImagePreprocessing.ImagePreprocessing(opt)
    intersection = IntersectionDetection.IntersectionDetection(opt, debug=True)
    im = cv2.resize(cv2.imread(in_dir), (width, height)) 
    # im = cv2.imread(in_dir)   
    im_cut = im[crop_height_value:, :]

    im_cut = processing_image.process_image2(im_cut)
    check_intersection = intersection.detect(im_cut)
    max_lines = check_intersection[1]['max_points']
    for i in max_lines:
        cv2.circle(im, (i[0], i[1] + crop_height_value), 1, (0, 0, 255), -1)
    print(check_intersection[0])
    cv2.imshow('image', im)
    cv2.imshow('a', im_cut)
    cv2.waitKey(0)
if __name__ == "__main__":
    test_lane_img(r'run_real/frame247.jpg')
    # test_1_img(r'run_real/frame247.jpg')