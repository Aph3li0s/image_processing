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
    ImageProcessor = ImagePreprocessing.ImagePreprocessing(opt)
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
            
            lane_det, grayIm = ImageProcessor.process_image(im)
            im_cut = im[crop_height_value:, :]
            hlane_det = ImageProcessor.process_image2(im_cut)
            check_intersection = intersection.detect(hlane_det)
            max_lines = check_intersection[1]['max_points']
            cv2.imwrite(os.path.join(intersection_img, path), im)
            if check_intersection[0][1]<= check_thresh['gap_thresh'] \
                and check_intersection[0][0]>= check_thresh['max_points_thresh']:
                print(image_path)
                cv2.putText(im, 'intersection', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
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
            lane_det, grayIm = ImageProcessor.process_image(im)            
            
            left_points, right_points, _, _ = LaneLine.find_left_right_lane(lane_det)
            speed, angle, state, lane_data = LaneKeeper.lane_keeping(left_points, right_points)
            left_points = lane_data["left_points"] 
            right_points = lane_data["right_points"] 
            left_point = lane_data["left_point"] 
            right_point = lane_data["right_point"] 
            middle_point = lane_data["middle_point"] 

            new_im = np.copy(im)
                
            if left_point == [0, 0]:
                cv2.putText(new_im, 'max right', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            elif right_point == [0, 0]:
                cv2.putText(new_im, 'max left', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                new_im = cv2.line(new_im, left_point, right_point, 255, 2)
                new_im = cv2.line(new_im, (width//2, height), middle_point, 255, 2)
            cv2.imwrite(os.path.join(lane_keep_img, path), new_im)
            if show:
                # cv2.imshow('raw image', im)
                cv2.imshow('lane keeping', new_im)
                
def process_video(in_vid, show = True):
    import logging
    logging.basicConfig(level=logging.INFO) 
    fps_limit = 15 #15
    opt = action.load_config_file("main_rc.json")
    check_thresh = opt['INTERCEPT_DETECTION']
    crop_ratio = float(check_thresh['crop_ratio'])
    height = opt["IMAGE_SHAPE"]["height"]
    width = opt["IMAGE_SHAPE"]["width"]
    crop_height_value =  int(height * crop_ratio)
    vid = cv2.VideoCapture(in_vid)
    
    ImageProcessor = ImagePreprocessing.ImagePreprocessing(opt)
    IntersectFinder = IntersectionDetection.IntersectionDetection(opt, debug=True)
    LaneLine = LaneDetection.LaneDetection(opt)
    LaneKeeper = LaneKeeping.LaneKeeping(opt, True)
    cnt_timer_intersect = 0
    cnt_timer_intersect_check = 0
    cnt_timer_curve = 0
    cnt_curve = 0
    cnt_intersect = 0
    intersect_timer = 0
    curve_timer = 0
    intersect_check = False
    max_curve = False
    blind_curve = False
    while True:
        ret, frame = vid.read()
        frame_resize = frame.copy()
        frame_resize = cv2.resize(frame_resize, (width, height))
        if not ret:
            print("Failed to capture frame")
            break
        blank_frame = np.zeros((50, 250, 3), dtype=np.uint8)
        speed_frame = np.zeros((150, 250, 3), dtype=np.uint8)
        frame_cut = frame_resize[crop_height_value:, :]
        lane_det, grayIm = ImageProcessor.process_image(frame_resize)
        # Intersection detect
        hlane_det = ImageProcessor.process_image2(frame_cut)
        check_intersection = IntersectFinder.detect(hlane_det)
        max_lines = check_intersection[1]['max_points']
        if check_intersection[0][1]<= check_thresh['gap_thresh'] \
            and check_intersection[0][0]>= check_thresh['max_points_thresh']:
                for i in max_lines:
                    cv2.circle(frame_resize, (i[0], i[1] + crop_height_value), 1, (0, 0, 255), -1)
                intersect_check = True

        left_points, right_points, _, _ = LaneLine.find_left_right_lane(lane_det)
        speed, angle, state, lane_data = LaneKeeper.lane_keeping(left_points, right_points)
        left_points = lane_data["left_points"] 
        right_points = lane_data["right_points"] 
        left_point = lane_data["left_point"] 
        right_point = lane_data["right_point"] 
        middle_point = lane_data["middle_point"] 

        frame_new = np.copy(frame_resize)
        cv2.putText(speed_frame, f'Speed: {round(speed, 3)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(speed_frame, f'Angle: {round(angle, 3)}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if left_point == [0, 0]:
            # cv2.putText(frame_new, 'max right', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            max_curve = True
            pass
        elif right_point == [0, 0]:
            # cv2.putText(frame_new, 'max left', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            max_curve = True
            pass
        else:
            frame_new = cv2.line(frame_new, left_point, right_point, 255, 2)
            frame_new = cv2.line(frame_new, (width//2, height), middle_point, 255, 2)
            max_curve = False
        
        # Normal case on lane
        if intersect_check == False:
            if max_curve ==  False:
                cv2.putText(blank_frame, "On lane", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if max_curve == True:
                cv2.putText(blank_frame, "Max curve", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  
        # Intersection case 
        if intersect_check == True:
            cnt_intersect += 1
            if cnt_intersect == 1:
                intersect_timer = time.time()
            # print(time.time() - intersect_timer)
            if time.time() - intersect_timer > 2:
                cnt_timer_intersect += 1
                if cnt_timer_intersect == 1:
                    cv2.imwrite('inter.jpg', frame)
                    # print('saved inter')
                    blind_curve = True
        # Blind curve case
        if blind_curve == True:
            cnt_curve += 1
            if left_point == [0, 0]:
                if cnt_curve == 1:
                    curve_timer = time.time()
                if time.time() - curve_timer <= 4:
                    cv2.putText(blank_frame, "Right blind curve", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
                    # print(time.time() - curve_timer)
                else:
                    cv2.imwrite('done_curve.jpg', frame)
                    # print('saved curve')
                    max_curve = False
                    blind_curve = False
                    intersect_check = False
                    cnt_curve = 0
                    cnt_intersect = 0
                    cnt_timer_intersect = 0
            elif right_point == [0, 0]:
                if cnt_curve == 1:
                    curve_timer = time.time()
                if time.time() - curve_timer <= 3:
                    cv2.putText(blank_frame, "Left blind curve", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
                else:
                    cv2.imwrite('done_curve.jpg', frame)
                    max_curve = False
                    blind_curve = False
                    intersect_check = False
                    cnt_curve = 0
                    cnt_intersect = 0
                    cnt_timer_intersect = 0

        if show:
            cv2.imshow('Raw Video', frame)
            cv2.imshow('Road Detection', cv2.resize(lane_det, (512, 384)))
            # cv2.imshow('Speed', speed_frame)
            cv2.imshow('Lane Keeping', frame_new)
            cv2.imshow('Move state', blank_frame)
        if cv2.waitKey(int(1000 / fps_limit)) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # process_image(in_img=r'run_real', intersection_img = r'intersection_img', lane_keep_img=None, show = None)
    process_video(in_vid=r'vid.avi')
