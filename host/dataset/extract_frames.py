import cv2
import os

def mov_to_photos(video_path, output_folder, frame_rate_skip=1):
    """
    Converts a video file into a sequence of image frames.

    :param video_path: Path to the input video file (e.g., '/Users/sherrylee/...).
    :param output_folder: Directory to save the extracted images.
    :param frame_rate_skip: Save one frame every 'n' frames (1 saves every frame).
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file at {video_path}")
        print("Please check the path and confirm FFmpeg is installed correctly.")
        return

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video file: {video_path}")
    print(f"Total Frames: {frame_count}")
    print(f"Frame Rate (FPS): {fps}")
    print(f"Frame Skip Rate: {frame_rate_skip}")
    print("-" * 30)

    current_frame = 0
    saved_frame_count = 0

    # Loop through all frames in the video
    while cap.isOpened():
        # Read the next frame
        ret, frame = cap.read()

        # If ret is False, we've reached the end of the video
        if not ret:
            break

        # Check if the frame should be saved based on the skip rate
        if current_frame % frame_rate_skip == 0:
            # Construct the filename: frame_00000.jpg, frame_00001.jpg, etc.
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:05d}.jpg")

            # Save the frame as a JPEG file
            cv2.imwrite(frame_filename, frame)

            saved_frame_count += 1
            if saved_frame_count % 100 == 0:
                 print(f"Processing... {saved_frame_count} frames saved.")

        current_frame += 1

    # Release the video capture object and clean up
    cap.release()
    cv2.destroyAllWindows()

    print("-" * 30)
    print(f"✅ Conversion complete!")
    print(f"Total frames processed: {current_frame}")
    print(f"Total images saved: {saved_frame_count} in '{output_folder}'")


# ----------------------------------------------------------------------
#                         *** CONFIGURATION ***
# ----------------------------------------------------------------------
# 1. THE CORRECTED PATH: Use the absolute path and remove all backslashes (\)
VIDEO_FILE = '/Users/sherrylee/Desktop/Master/Master 3-1/Embedded System/lab_3/Al1106.MOV' 

# 2. The output folder will be created in the same directory where you run the script
OUTPUT_DIR = 'video_frames_outputAlbert1106' 

# 3. Frame Skip Rate: 
#    1 = Saves EVERY frame (large output!)
#    30 = Saves about one frame per second (if video is 30 FPS)
SKIP_RATE = 1 

# Run the function
mov_to_photos(VIDEO_FILE, OUTPUT_DIR, SKIP_RATE)