import cv2
import numpy as np
import argparse
import os
import concurrent.futures
import ffmpeg

def process_frame_cpu(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to smoothen the image
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply median blur to reduce noise
    gray = cv2.medianBlur(gray, 7)

    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # Apply a cartoon effect by combining the edges and the original frame
    cartoon = cv2.bitwise_and(frame, frame, mask=edges)

    return cartoon

def process_frame_gpu(frame_gpu):
    # Convert the frame to grayscale
    gray = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to smoothen the image
    gray = cv2.cuda.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply median blur to reduce noise
    gray = cv2.cuda.medianBlur(gray, 7)

    # Detect edges using adaptive thresholding
    edges = cv2.cuda.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # Apply a cartoon effect by combining the edges and the original frame
    cartoon = cv2.cuda.bitwise_and(frame_gpu, frame_gpu, mask=edges)

    return cartoon

def cartoonify_video(input_file, output_file=None, verbose=False, use_gpu=False):
    def print_verbose(message):
        if verbose:
            print(message)

    # Open the video file using FFmpeg
    input_video = ffmpeg.input(input_file)

    # Create FFmpeg process for writing the output video with the same encoding
    base_filename, file_extension = os.path.splitext(input_file)
    if output_file is None:
        output_file = f"{base_filename}_cartoon{file_extension}"

    output_video = (
        ffmpeg.output(
            input_video.video,
            output_file,
            vf='format=yuv420p'
        )
        .overwrite_output()
    )

    # Run the FFmpeg process
    ffmpeg.run(output_video, quiet=True, overwrite_output=True)

    # Initialize video capture using OpenCV
    cap = cv2.VideoCapture(input_file)

    frame_queue = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_queue.append(frame)

    def process_frame_with_queue(frame):
        if use_gpu:
            frame_gpu = cv2.cuda_GpuMat()
            frame_gpu.upload(frame)
            cartoon_gpu = process_frame_gpu(frame_gpu)
            cartoon = cartoon_gpu.download()
        else:
            cartoon = process_frame_cpu(frame)

        return cartoon

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_frame_with_queue, frame_queue))

    # Release video objects
    cap.release()

    print_verbose("Cartoonification completed.")

    # Write the cartoon frames to the output video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, cap.get(5), (int(cap.get(3)), int(cap.get(4))))

    for cartoon in results:
        out.write(cartoon)

    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cartoonify a video file.")
    parser.add_argument("input_file", help="Input video file name")
    parser.add_argument("-o", "--output_file", help="Output video file name (optional)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for acceleration if available")
    args = parser.parse_args()

    cartoonify_video(args.input_file, args.output_file, args.verbose, args.gpu)
