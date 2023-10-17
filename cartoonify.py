import cv2
import numpy as np
import argparse
import os
import concurrent.futures

def cartoonify_video(input_file, output_file=None, verbose=False, use_gpu=False):
    def print_verbose(message):
        if verbose:
            print(message)

    # Open the video file
    cap = cv2.VideoCapture(input_file)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Get audio properties
    audio_enabled = False
    audio_fps = 0
    audio_frame_size = 0
    audio_codec = None
    audio_writer = None

    if cv2.haveVideoWriter_ffmpeg():
        audio_fps = int(cap.get(cv2.CAP_PROP_FPS))
        audio_frame_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        audio_codec = cv2.VideoWriter_fourcc(*'mp4v')
        audio_enabled = True

        if output_file is None:
            base_filename, file_extension = os.path.splitext(input_file)
            output_file = f"{base_filename}_cartoon{file_extension}"

        audio_writer = cv2.VideoWriter(output_file, audio_codec, audio_fps, frame_size)

    # Define the codec and create a VideoWriter object for the cartoon video
    if output_file is None:
        base_filename, file_extension = os.path.splitext(input_file)
        output_file = f"{base_filename}_cartoon{file_extension}"

    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

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
        if audio_enabled:
            audio_writer.write(cartoon)

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if audio_enabled:
        audio_writer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cartoonify a video file.")
    parser.add_argument("input_file", help="Input video file name")
    parser.add_argument("-o", "--output_file", help="Output video file name (optional)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for acceleration if available")
    args = parser.parse_args()

    cartoonify_video(args.input_file, args.output_file, args.verbose, args.gpu)
