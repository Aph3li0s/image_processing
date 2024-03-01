import socket
import pickle
from PIL import Image
from io import BytesIO
import cv2
import struct
import os

try:
    os.mkdir("./captures/")
except:
    pass

# Kết nối đến server
server_address = ('192.168.7.232', 1234)  # Địa chỉ và cổng của server
client_socket = socket.socket()
client_socket.connect(server_address)

payload_size = struct.calcsize("Q")

# Nhận dữ liệu từ server

data = b""
count = 0
frame_count = 0
img_array = []
while True:
    chunk = client_socket.recv(4*1024)
    if not chunk:
        break
    data+=chunk
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]
    
    while len(data)<msg_size:
        data+=client_socket.recv(4*1024)
    image = data[:msg_size]
    data = data[msg_size:]
    
    # print(len(image))
    image = pickle.loads(image)
    # print(image)
    # image = cv2.resize(image,(640,480))
    cv2.imshow('frame', image)
    cv2.imwrite(f"./captures/frame{frame_count}.jpg", image)
    # cv2.waitKey(0)
    # if (count % 5 == 0):
    #     frame_count += 1
    #     cv2.imwrite(f"./captures/frame{frame_count}.jpg", image)
    # img_array.append(image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # out = cv2.VideoWriter('./captures/capture.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (640,480))
        # for i in img_array:
        #     out.write(i)
        # out.release()
        break
    
    count += 1
cv2.destroyAllWindows()

# Đóng kết nối
client_socket.close()