import cv2
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread(r'captures/frame54.jpg')

# Convert BGR image to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(img_rgb)
plt.axis('on')  # Turn on axis
plt.show()
