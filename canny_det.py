import time
import numpy as np
import os 
import cv2
import matplotlib.pyplot as plt

class ImagePreprocessing():
    def get_sobel_image(self, sobel_image, threshold):
        abs_sobel_image = np.absolute(sobel_image)
        scaled_sobelx = np.uint8(255*abs_sobel_image/np.max(abs_sobel_image))
        binary = np.zeros_like(scaled_sobelx)
        binary[(scaled_sobelx >= threshold[0]) 
                & (scaled_sobelx <= threshold[1])] = 255
        
        return binary

    def process_image(self, frame):
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        bgr_image = np.copy(frame)
        
        bgr_image = self.region_of_interest(bgr_image)
        red_channel = bgr_image[:,:,2]
        hls = np.float64(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HLS))
        l_channel = hls[:,:,1]

        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize = 3)
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize = 3)

        sxbinary = self.get_sobel_image(sobelx, [25, 255])
        sybinary = self.get_sobel_image(sobely, [110, 250])

        r_binary = np.zeros_like(red_channel)
        r_binary[(red_channel >= 60) & (red_channel <= 255)] = 255
        combined_binary = np.zeros_like(sxbinary)
        
        grayImg = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        # _, image_ff = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        xy_bin = sxbinary + sybinary
        grayImg[grayImg >= 17] = 255
        adaptive = cv2.adaptiveThreshold(grayImg, 255,\
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, -50)
        combined_binary[((adaptive == 255) & (sxbinary == 255))
                        | ((sxbinary == 255) & (r_binary == 255))] = 255
        
        new_combined_binary = combined_binary
        new_combined_binary[((combined_binary == 255) & (adaptive == 255))] = 255
        new_combined_binary = self.region_of_interest(new_combined_binary)
        
        new_combined_binary = cv2.dilate(new_combined_binary, \
                                np.ones((3, 3), np.uint8)) 

        new_combined_binary = cv2.erode(new_combined_binary, \
                                np.ones((3,3), np.uint8)) 
        

        return new_combined_binary
    
    
    def region_of_interest(self, frame):
        height = frame.shape[0]
        width = frame.shape[1]
        mask = np.zeros_like(frame)

        # region_of_interest_vertices = np.array([[   (0, height),
        #                                             (0, height - 50),
        #                                             (0.25 * width, 0.4 * height),
        #                                             (0.65 * width, 0.4 * height),
        #                                             (width, height-50),
        #                                             (width, height)]], np.int32)
        region_of_interest_vertices = np.array([[   (0, height),
                                                    (width, height),
                                                    (width, height - 60),
                                                    (width - 200, 0.42 * height),
                                                    (200, 0.42 * height),
                                                    (0, height - 60),
                                                    (0, height)]], np.int32)
        cv2.fillPoly(mask, region_of_interest_vertices, 255)
        masked_image = cv2.bitwise_and(frame, mask)
        return masked_image
      
if __name__ == '__main__':
    im_dir = r'test_im'
    save_dir = r'save_im'
    im_pros = ImagePreprocessing()
    for path in os.listdir(im_dir):
        image_path = os.path.join(im_dir, path)
        im = cv2.imread(image_path)    
        result = im_pros.process_image(im)
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(save_dir, path), result)
        # plt.figure()
        # plt.imshow(result)    
        # save_path = os.path.join(save_dir, path)
        # plt.imsave(save_path, result)