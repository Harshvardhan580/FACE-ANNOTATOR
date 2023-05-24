import argparse
import os
import glob
import cv2
from sfd_detector import SFDDetector

def extract_faces_from_videos(input_dir, output_dir):
    device = 'cuda'
    detector = SFDDetector(device, path_to_detector='s3fd.pth')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_file in glob.glob(os.path.join(input_dir, '*.mp4')):
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        video_output_dir = os.path.join(output_dir, video_name)

        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        cap = cv2.VideoCapture(video_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(frame_count):
            ret, frame = cap.read()

            if not ret:
                break

            bboxes = detector.detect_from_image(frame)

            for j, bbox in enumerate(bboxes):
                x1, y1, x2, y2, score = bbox
                face = frame[int(y1):int(y2), int(x1):int(x2)]

                if face.size == 0:
                    continue

                output_path = os.path.join(video_output_dir, f'face_{i}_{j}.jpg')
                cv2.imwrite(output_path, face)

        cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="path to input directory containing videos", type=str)
    parser.add_argument("--output_dir", help="path to output directory for extracted faces", type=str)
    args = parser.parse_args()

    if not args.input_dir:
        print("Please provide path to input directory containing videos.")
        return
    if not args.output_dir:
        print("Please provide path to output directory for extracted faces.")
        return

    extract_faces_from_videos(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()
