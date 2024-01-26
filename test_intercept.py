import ImagePreprocessing
import InterceptDetection 
import time
import numpy as np
import os 
import cv2
import utils.utils_action as action


if __name__ == "__main__":
    im_dir = r'test_real' 
    save_dir_y = r'save_im3'
    opt = action.load_config_file("main_rc.json")
    im_pros = ImagePreprocessing.ImagePreprocessing(opt)
    intercept = InterceptDetection.InterceptDetection(opt, debug=True)
    
    for path in os.listdir(save_dir_y):
        image_path = os.path.join(save_dir_y, path)
        if os.path.isfile(image_path):
            os.remove(image_path)
    for path in os.listdir(im_dir):
        key = cv2.waitKey(0)
        if key == ord('e'):
            break
        image_path = os.path.join(im_dir, path)
        im = cv2.imread(image_path)    
        im_cut = im[int(480 * 0.35):, :]
        res = im_pros.process_image(im)
        result = im_pros.process_image2(im_cut)
        height = int(im.shape[0] * float(opt['INTERCEPT_DETECTION']['crop_ratio']))
        cv2.imshow('raw image', res)
        cv2.imwrite(os.path.join(save_dir_y, path), im)
        print(height)
        check_intercept = intercept.detect(result)
        max_lines = check_intercept[1]['max_points']
        if check_intercept[0][1]<= 120:
            if check_intercept[0][0]>= 240:
                # print(path)
                for i in max_lines:
                    cv2.circle(im, (i[0], i[1] + height), 2, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(save_dir_y, path), im)
        cv2.imshow('intersection', im)