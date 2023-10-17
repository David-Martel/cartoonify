import cv2
import numpy as np
import argparse
import os
import concurrent.futures

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

    # Check if CUDA is available and use it for acceleration if specified
    if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print_verbose("Using GPU for acceleration.")
        cv2.cuda.printCudaDeviceInfo(0)
        cv2.cuda.setDevice(0)
    else:
        use_gpu = False
        print_verbose("GPU not available or not specified. Using CPU for processing.")

    # Open the video file
    cap = cv2.VideoCapture(input_file)

    # Define the codec and create a VideoWriter object
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + "_cartoon" + os.path.splitext(input_file)[1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

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

    for cartoon in results:
        out.write(cartoon)

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cartoonify a video file.")
    parser.add_argument("input_file", help="Input video file name")
    parser.add_argument("-o", "--output_file", help="Output video file name (optional)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for acceleration if available")
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit")
    args = parser.parse_args()

    cartoonify_video(args.input_file, args.output_file, args.verbose, args.gpu)
