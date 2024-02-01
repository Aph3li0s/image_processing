import matplotlib.pyplot as plt
import cv2
import utils.utils_action as action

im = cv2.imread(r'run_real/frame92.jpg')
im = cv2.resize(im, (240, 320))
opt = action.load_config_file("main_rc.json")
height = opt["IMAGE_SHAPE"]["height"]
width = opt["IMAGE_SHAPE"]["width"]
check_thresh = opt['INTERCEPT_DETECTION']
crop_ratio = float(check_thresh['crop_ratio'])
crop_height_value =  int(height * crop_ratio)
im_cut = im[crop_height_value:, :]
image_rgb = cv2.cvtColor(im_cut, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.show()