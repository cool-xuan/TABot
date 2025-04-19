import cv2
import os
import json
from pathlib import Path

def extract_frames_from_video(json_file_path, video_folder, output_image_folder):

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    video_path = os.path.join(video_folder, data['videoPath'])
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    for segment in data['accident_objects']:
        timestamp = float(segment['timestamp'])
        frame_position = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = cap.read()

        if ret:
            base_name = data['videoPath'].replace("videos/", "").replace(".mp4", "")
            image_name = f"{base_name}_{timestamp:.3f}.jpg"
            output_path = os.path.join(output_image_folder, image_name)
            
            cv2.imwrite(output_path, frame)
            print(f"Extracted frame {output_path}")

    cap.release()

def process_json_files_in_folder(json_folder, video_folder, output_image_folder):
    
    pathlist = Path(json_folder).glob('**/*.json')
    for path in pathlist:
        json_file_path = str(path)
        extract_frames_from_video(json_file_path, video_folder, output_image_folder)

json_folder = "/path/to/video_annotations/" 
base_folder = "/path/to/your/folder/"  
output_image_folder = base_folder + "images/"  

process_json_files_in_folder(json_folder, base_folder, output_image_folder)