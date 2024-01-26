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
        cv2.imwrite(os.path.join(out_img, path), im)
        
        if check_intercept[0][1]<= check_thresh['gap_thresh'] \
            and check_intercept[0][0]>= check_thresh['max_points_thresh']:
                # print(path)
                for i in max_lines:
                    cv2.circle(im, (i[0], i[1] + crop_height_value), 1, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(out_img, path), im)
                
        if show:
            # show lane
            cv2.imshow('raw image', lane_det)
            cv2.imshow('intersection', im)

def process_video(in_vid, out_dir, show = True):
    opt = action.load_config_file("main_rc.json")
    im_pros = ImagePreprocessing.ImagePreprocessing(opt)
    intercept = InterceptDetection.InterceptDetection(opt, debug=True)
    
    vid = cv2.VideoCapture(in_vid)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    cnt = 0
    while True:
        ret, frame = vid.read()
        frame = cv2.resize(frame, (640, 480))
        if not ret:
            print("Failed to capture frame")
            break
        check_thresh = opt['INTERCEPT_DETECTION']
        crop_ratio = float(check_thresh['crop_ratio'])
        height = frame.shape[0]
        crop_height_value =  int(height * crop_ratio)
        
        frame_cut = frame[crop_height_value:, :]
        lane_det = im_pros.process_image(frame)
        hlane_det = im_pros.process_image2(frame_cut)
        check_intercept = intercept.detect(hlane_det)
        
        max_lines = check_intercept[1]['max_points']
        if check_intercept[0][1]<= check_thresh['gap_thresh'] \
            and check_intercept[0][0]>= check_thresh['max_points_thresh']:
                for i in max_lines:
                    cv2.circle(frame, (i[0], i[1] + crop_height_value), 1, (0, 0, 255), -1)
        if show:
            cv2.imshow('Raw Video', cv2.resize(frame, (512, 384)))
            cv2.imshow('Road Detection', cv2.resize(lane_det, (512, 384)))
            cv2.imshow('Intersection', hlane_det)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_image(in_img=r'test_real', out_img = r'save_im3', show = False)
    # process_video(in_vid=r'test_vid/crossroad1.mp4', out_dir=r'save_im2')
