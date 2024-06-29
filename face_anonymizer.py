import os
import cv2
import time
import imageio
from tqdm import tqdm
import argparse
from utils import anonymize_face
from model.centerface import CenterFace

os.makedirs('output', exist_ok=True)

def perform_face_anonymization(
      input_video_path,
      output_dir='output',
      model_name='centerface',
      threshold=0.20,
      keep_audio=True,
      mask_scale=1.2,
      replacewith='blur',
      ellipse=True,
    ):
    print("\nProcessing - ", input_video_path, "\n")
    output_video_path = os.path.join(output_dir, input_video_path.split('/')[-1].replace('.mp4', '_anonymized.mp4'))

    if model_name == 'centerface':
        centerface = CenterFace(backend='auto')

    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    _ffmpeg_config = {"codec": "libx264"}
    _ffmpeg_config.setdefault('fps', fps)
    reader = imageio.get_reader(input_video_path)
    meta = reader.get_meta_data()

    if keep_audio and meta.get('audio_codec'):
        _ffmpeg_config.setdefault('audio_path', input_video_path)
        _ffmpeg_config.setdefault('audio_codec', 'copy')
    writer = imageio.get_writer(output_video_path, format='FFMPEG', mode='I', **_ffmpeg_config)

    frame_num = 0
    start_time = time.time()
    previous_face_detection = []

    # Read video frame by frame and perform anonymization
    with tqdm(total=frame_count) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_detections, _ = centerface(frame, threshold=threshold)
            if len(face_detections) != 0:
                previous_face_detection.append(face_detections)
                anonymize_face(face_detections=face_detections, frame=frame,
                               mask_scale=mask_scale,
                               model_name=model_name)
            if len(face_detections) == 0 and len(previous_face_detection) != 0:
                anonymize_face(face_detections=previous_face_detection[-1], frame=frame,
                               mask_scale=mask_scale,
                               model_name=model_name)
            writer.append_data(frame)
            frame_num += 1
            pbar.update(1)
        writer.close()
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform face anonymization on a video.')
    parser.add_argument('input_video_path', type=str, help='Path to the input video file.')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the output video.')
    parser.add_argument('--model_name', type=str, default='centerface', help='Face detection model to use.')
    parser.add_argument('--threshold', type=float, default=0.20, help='Threshold for face detection.')
    parser.add_argument('--keep_audio', type=bool, default=True, help='Whether to keep the audio from the original video.')
    parser.add_argument('--mask_scale', type=float, default=1.2, help='Scale of the anonymization mask.')
    parser.add_argument('--replacewith', type=str, default='blur', help='Type of anonymization (e.g., blur).')
    parser.add_argument('--ellipse', type=bool, default=True, help='Whether to use an elliptical mask for anonymization.')

    args = parser.parse_args()
    perform_face_anonymization(
        input_video_path=args.input_video_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        threshold=args.threshold,
        keep_audio=args.keep_audio,
        mask_scale=args.mask_scale,
        replacewith=args.replacewith,
        ellipse=args.ellipse,
    )
