import cv2 
import numpy as np 
import os
import matplotlib.pyplot as plt

im_dir = r'test_im'
save_dir = r'save_im'

def threshold_filter(sobel_image, threshold):
    abs_sobel_image = np.absolute(sobel_image)
    scaled_sobelx = np.uint8(255*abs_sobel_image/np.max(abs_sobel_image))
    binary = np.zeros_like(scaled_sobelx)
    binary[(scaled_sobelx >= threshold[0]) & (scaled_sobelx <= threshold[1])] = 255
    return binary

def region_of_interest(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    mask = np.zeros_like(frame)

    region_of_interest_vertices = np.array([[   (200, 0.5 * height),   
                                                (100, height),
                                                (0, height),
                                                (0, 0.5 * height),
                                                (200, 0.5 * height),
                                                
                                                (width - 200, 0.5 * height),
                                                (width - 100, height),
                                                (width, height),
                                                (width, 0.5 * height),
                                                (width - 200, 0.5 * height)]], np.int32)
    # region_of_interest_vertices = np.array([[   (0, height),
    #                                             (width, height),
    #                                             (width, height - 100),
    #                                             (width - 200, 0.5 * height),
    #                                             (200, 0.5 * height),
    #                                             (0, height - 100),
    #                                             (0, height)]], np.int32)
    
    cv2.fillPoly(mask, region_of_interest_vertices, 255)
    
    masked_image = cv2.bitwise_and(frame, mask)
    return mask

def sobel_image(image):
    im = cv2.GaussianBlur(image,(3,3),0)
    grey = np.float64(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

    sobel_x = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize = 3)
    sobel_y = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize = 3)
    sxbinary = threshold_filter(np.hypot(sobel_x, sobel_y), [25, 255])
    ret, thresh = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY)
    # ret = sxbinary + thresh
    return sxbinary

def hough_transform(image):
    hough_lines = cv2.HoughLinesP(image, rho = 5, theta = np.pi/180, threshold = 10, 
                    minLineLength = 20, maxLineGap =500)
    
    left_lane, right_lane = average_slope_intercept(hough_lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def average_slope_intercept(lines):
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # calculating slope of a line
            slope = (y2 - y1) / (x2 - x1)
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            # slope of left lane is negative and for right lane slope is positive
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    # 
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=6):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

if __name__ == '__main__':
    for path in os.listdir(im_dir):
        image_path = os.path.join(im_dir, path)
        im = cv2.imread(image_path)    
        bin_sobel = sobel_image(im)
        hough = hough_transform(region_of_interest(bin_sobel))
        print(hough)
        result = draw_lane_lines(region_of_interest(bin_sobel), hough)
        plt.figure()
        # plt.imshow(bin_sobel)
        # result = region_of_interest(bin_sobel)
        plt.imshow(result)
        # if path == '77.jpg':
        #     plt.show()      
        save_path = os.path.join(save_dir, path)
        plt.imsave(save_path, result)
        # cv2.imshow("masked", region_of_interest(bin_sobel))
        # cv2.imshow("result", result)
        # cv2.waitKey(0)