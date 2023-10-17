import cv2
import argparse
import os

def snip_video_time(input_file, start_time, stop_time, output_file=None):
    # Open the video file
    cap = cv2.VideoCapture(input_file)

    # Get the frame rate of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate the frame indices for the start and stop times
    start_frame = int(start_time * fps)
    stop_frame = int(stop_time * fps)

    # Check if the specified start and stop times are within the video duration
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame >= stop_frame or start_frame < 0 or stop_frame > total_frames:
        print("Invalid start or stop time.")
        return

    # Read the video frames and write the snippet to an output file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if output_file is None:
        base_filename, file_extension = os.path.splitext(input_file)
        output_file = f"{base_filename}_t-{start_time}_t-{stop_time}{file_extension}"

    out = cv2.VideoWriter(output_file, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while cap.get(cv2.CAP_PROP_POS_FRAMES) <= stop_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snip a portion of a video.")
    parser.add_argument("input_file", help="Input video file name")
    parser.add_argument("start_time", type=float, help="Start time in seconds")
    parser.add_argument("stop_time", type=float, help="Stop time in seconds")
    parser.add_argument("-o", "--output_file", help="Output video file name (optional)")
    args = parser.parse_args()

    snip_video_time(args.input_file, args.start_time, args.stop_time, args.output_file)
