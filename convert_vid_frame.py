import cv2

def extract_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through each frame and save it as an image
    for frame_number in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_number}")
            break

        # Save the frame as an image
        frame_filename = f"{output_folder}/frame{frame_number+1}.jpg"
        cv2.imwrite(frame_filename, frame)

        print(f"Frame {frame_number}/{total_frames} saved: {frame_filename}")

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    video_path = "captures/capture.avi"  # Change this to your video file path
    output_folder = "run_real"  # Change this to the folder where you want to save frames

    extract_frames(video_path, output_folder)
