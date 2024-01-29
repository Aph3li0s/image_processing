import ImagePreprocessing
import IntersectionDetection
import LaneDetection
import LaneKeeping
import time
import numpy as np
import os 
import cv2
import utils.utils_action as action

def clear_folder(in_folder):
    for path in os.listdir(in_folder):
        image_path = os.path.join(in_folder, path)
        if os.path.isfile(image_path):
            os.remove(image_path)
            
def display_points(points, image):
    if points is not None:
        for point in points:
            image = cv2.circle(image, point, 1, (0, 0, 255), -1)    #red
    return image

def process_image(in_img, intersection_img, lane_keep_img, show):
    opt = action.load_config_file("main_rc.json")
    processing_image = ImagePreprocessing.ImagePreprocessing(opt)
    check_thresh = opt['INTERCEPT_DETECTION']
    crop_ratio = float(check_thresh['crop_ratio'])
    height = opt["IMAGE_SHAPE"]["height"]
    width = opt["IMAGE_SHAPE"]["width"]
    crop_height_value =  int(height * crop_ratio)
    
    if intersection_img is not None:
        clear_folder(intersection_img)
        intersection = IntersectionDetection.IntersectionDetection(opt, debug=True)
        for path in os.listdir(in_img):
            key = cv2.waitKey(0)
            if key == ord('e'):
                break
            image_path = os.path.join(in_img, path)
            im = cv2.resize(cv2.imread(image_path), (width, height))  
            
            lane_det = processing_image.process_image(im)
            im_cut = im[crop_height_value:, :]
            hlane_det = processing_image.process_image2(im_cut)
            check_intersection = intersection.detect(hlane_det)
            max_lines = check_intersection[1]['max_points']
            cv2.imwrite(os.path.join(intersection_img, path), im)
            
            if check_intersection[0][1]<= check_thresh['gap_thresh'] \
                and check_intersection[0][0]>= check_thresh['max_points_thresh']:
                    # print(path)
                    for i in max_lines:
                        cv2.circle(im, (i[0], i[1] + crop_height_value), 1, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(intersection_img, path), im)
                
    if lane_keep_img is not None:
        clear_folder(lane_keep_img)
        LaneLine = LaneDetection.LaneDetection(opt)
        LaneKeeper = LaneKeeping.LaneKeeping(opt, True)
        for path in os.listdir(in_img):
            key = cv2.waitKey(0)
            if key == ord('e'):
                break
            image_path = os.path.join(in_img, path)
            im = cv2.resize(cv2.imread(image_path), (width, height))    
            lane_det = processing_image.process_image(im)            
            
            left_points, right_points, _, _ = LaneLine.find_left_right_lane(lane_det)
            speed, angle, state, lane_data = LaneKeeper.lane_keeping(left_points, right_points)
            left_points = lane_data["left_points"] 
            right_points = lane_data["right_points"] 
            left_point = lane_data["left_point"] 
            right_point = lane_data["right_point"] 
            middle_point = lane_data["middle_point"] 

            new_im = np.copy(im)
                
            new_im = cv2.line(new_im, left_point, right_point, 255, 2)
            new_im = cv2.line(new_im, (height, width), middle_point, 255, 2)
            cv2.imwrite(os.path.join(lane_keep_img, path), new_im)
                # print('speed: ', speed)
                # print('angle: ', angle)
                # print('state: ', state)
                # print('')
            if show:
                # show lane
                # cv2.imshow('raw image', im)
                cv2.imshow('lane keeping', new_im)

def process_video(in_vid, show = True):
    opt = action.load_config_file("main_rc.json")
    processing_image = ImagePreprocessing.ImagePreprocessing(opt)
    intersection = IntersectionDetection.IntersectionDetection(opt, debug=True)
    
    vid = cv2.VideoCapture(in_vid)
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
        lane_det = processing_image.process_image(frame)
        hlane_det = processing_image.process_image2(frame_cut)
        check_intersection = intersection.detect(hlane_det)
        
        max_lines = check_intersection[1]['max_points']
        if check_intersection[0][1]<= check_thresh['gap_thresh'] \
            and check_intersection[0][0]>= check_thresh['max_points_thresh']:
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
    process_image(in_img=r'test_real', intersection_img = r'intersection_img', lane_keep_img=r'lane_keeping_img', show = None)
    # process_video(in_vid=r'test_vid/crossroad1.mp4')
