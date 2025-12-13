import cv2
import os
import sys

def extract_frames(video_path, output_folder, fps_to_extract=None):
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get the total frame count, FPS, and duration of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)  # Correctly get the frames per second of the video
    duration = total_frames / fps_video  # Calculate video duration in seconds

    print(f"Video Duration: {duration:.2f} seconds")

    if fps_to_extract is None:
        print(f"Extracting all frames ({total_frames} frames).")
        interval = 1  # No interval, extract every frame
    else:
        print(f"Extracting {fps_to_extract} frame(s) per second.")
        interval = fps_video / fps_to_extract  # Calculate interval between frames
        print(f"Interval between frames: {interval:.2f} frames.")

    frame_count = 0
    extracted_frame_count = 0

    while True:
        # Set the frame position based on the time
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame if it matches the interval
        if fps_to_extract is None or frame_count % interval < 1:
            frame_filename = os.path.join(output_folder, f"frame_{extracted_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_frame_count += 1
            print(f"Saved {frame_filename}")

        # Increment frame count
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {extracted_frame_count} frames from the video.")

if __name__ == "__main__":
    # Check if correct number of arguments is provided
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python extract_frames.py <video_path> <output_folder> [<fps_to_extract>]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_folder = sys.argv[2]
    
    # If fps_to_extract is provided, parse it as an integer; otherwise, set it to None
    fps_to_extract = None
    if len(sys.argv) == 4:
        fps_to_extract = int(sys.argv[3])

    extract_frames(video_path, output_folder, fps_to_extract)
