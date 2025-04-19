import json
import subprocess
import os

base_video_path = '/path/to/base/videos/'  
split_video_path = '/path/to/split/videos/'     
json_path = '/path/to/your/video_bilibili.json'
os.makedirs(split_video_path, exist_ok=True)

with open(json_path, 'r') as file:
    data = json.load(file)

for main_video, info in data.items():
    main_video_path = os.path.join(base_video_path, main_video)

    for video in info['videos']:
        short_video_name = video['short_video']
        start_time = video['video_segment'][0]
        end_time = video['video_segment'][1]

        duration = end_time - start_time

        split_file_path = os.path.join(split_video_path, short_video_name)

        command = [
            'ffmpeg',
            '-i', main_video_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'copy',
            '-c:a', 'copy',
            split_file_path
        ]

        print(f"Extracting {short_video_name} from {main_video_path} (start: {start_time}, duration: {duration})")
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully extracted {short_video_name}")
        else:
            print(f"Failed to extract {short_video_name}: {result.stderr}")