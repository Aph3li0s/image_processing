import cv2
import os

frames_directory = "gray_img"
output_video_filename = "intersection_det.avi"
frame_files = [f for f in os.listdir(frames_directory) if f.endswith(".jpg")]
first_frame = cv2.imread(os.path.join(frames_directory, frame_files[0]))
height, width, _ = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video_writer = cv2.VideoWriter(output_video_filename, fourcc, 10.0, (width, height))
num_frames = len(os.listdir(frames_directory))
# Write each frame to the video
for i in range(50, num_frames):
    frame_path = os.path.join(frames_directory, f"frame{i+1}.jpg")
    frame = cv2.imread(frame_path)
    video_writer.write(frame)
# Release the video writer
video_writer.release()

print(f"Video created: {output_video_filename}")
