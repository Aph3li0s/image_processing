import ImagePreprocessing
import IntersectionDetection as IntersectionDetection 
import LaneKeeping
import time
import numpy as np
import os 
import cv2
import utils.utils_action as action

def test_1_img(in_dir):
    opt = action.load_config_file("main_rc.json")
    im_pros = ImagePreprocessing.ImagePreprocessing(opt)
    intercept = IntersectionDetection.InterceptDetection(opt, debug=True)
    
    im = cv2.imread(in_dir)
    check_thresh = opt['INTERCEPT_DETECTION']
    crop_ratio = float(check_thresh['crop_ratio'])
    height = im.shape[0]
    crop_height_value =  int(height * crop_ratio)
    
    im_cut = im[crop_height_value:, :]
    hlane_det = im_pros.process_image2(im_cut)
    check_intercept = intercept.detect(hlane_det)
    gap = check_intercept[0][1]
    print('gap value: ', gap)
    max_lines = check_intercept[1]['max_points']
    
    if check_intercept[0][1]<= check_thresh['gap_thresh'] and check_intercept[0][0]>= check_thresh['max_points_thresh']:
        for i in max_lines:
            cv2.circle(im, (i[0], i[1] + crop_height_value), 1, (0, 0, 255), -1)
    cv2.imshow('raw image', hlane_det)
    cv2.imshow('intersection', im)
    cv2.waitKey(0)

if __name__ == "__main__":
    test_1_img(r'test_real/image35.jpg')
