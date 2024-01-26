import ImagePreprocessing
import InterceptDetection 
import time
import numpy as np
import os 
import cv2
import utils.utils_action as action

def process_image(in_img, out_img, show):
    opt = action.load_config_file("main_rc.json")
    im_pros = ImagePreprocessing.ImagePreprocessing(opt)
    intercept = InterceptDetection.InterceptDetection(opt, debug=True)
    
    for path in os.listdir(out_img):
        image_path = os.path.join(out_img, path)
        if os.path.isfile(image_path):
            os.remove(image_path)
            
    for path in os.listdir(in_img):
        key = cv2.waitKey(0)
        if key == ord('e'):
            break
        
        image_path = os.path.join(in_img, path)
        im = cv2.imread(image_path)    
        check_thresh = opt['INTERCEPT_DETECTION']
        crop_ratio = float(check_thresh['crop_ratio'])
        height = im.shape[0]
        crop_height_value =  int(height * crop_ratio)
        
        im_cut = im[crop_height_value:, :]
        lane_det = im_pros.process_image(im)
        hlane_det = im_pros.process_image2(im_cut)
        
        check_intercept = intercept.detect(hlane_det)
        max_lines = check_intercept[1]['max_points']
        if show:
            cv2.imshow('raw image', lane_det)
            cv2.imshow('intersection', im)
        cv2.imwrite(os.path.join(out_img, path), im)
        
        if check_intercept[0][1]<= check_thresh['gap_thresh']:
            if check_intercept[0][0]>= check_thresh['max_points_thresh']:
                # print(path)
                for i in max_lines:
                    cv2.circle(im, (i[0], i[1] + crop_height_value), 2, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(out_img, path), im)

def process_video():
    pass

if __name__ == '__main__':
    process_image(in_img=r'test_real', out_img = r'save_im3', show = False)
