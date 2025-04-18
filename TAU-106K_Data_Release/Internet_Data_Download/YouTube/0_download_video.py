import json
import subprocess
import os

download_path = '/path/to/your/directory'
json_path = '/path/to/your/video_youtube.json'
os.makedirs(download_path, exist_ok=True)

with open(json_path, 'r') as file:
    data = json.load(file)

youtube_ids = list(data.keys())

for youtube_id in youtube_ids:
    output_filename = f"{youtube_id}.mp4"
    full_output_path = os.path.join(download_path, output_filename)
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    
    command = [
        'yt-dlp',
        '-o', full_output_path,
        '--format', 'bestvideo/best',
        '--no-audio',
        '--continue',
        url
    ]
    
    print(f"Downloading {url} to {full_output_path}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Successfully downloaded {full_output_path}")
    else:
        print(f"Failed to download {full_output_path}: {result.stderr}")