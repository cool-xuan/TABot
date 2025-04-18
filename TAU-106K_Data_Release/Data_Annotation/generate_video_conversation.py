import os
import json
import numpy as np
import random

random_seed = 9826
random.seed(random_seed)
np.random.seed(random_seed)

input_dir = '/path/to/video_annotations/'
video_path = '/path/to/your/videos/'
output_dir = 'tasks_epoch3'
generation_freq = 3
generate_segment_ref=True
generate_object_bbox=True

annotation_template = 'video_annotations_all_{}.json'.format

recognition_description_save_dir_name = 'recognition_description_video'
recognition_description_save_dir = os.path.join(output_dir, recognition_description_save_dir_name); os.makedirs(recognition_description_save_dir, exist_ok=True)
rd_converstaion_template = 'rd_video_conversations_all_{}.json'.format
rd_generation_freq = generation_freq

temporal_localization_save_dir_name = 'temporal_localization'
temporal_localization_save_dir = os.path.join(output_dir, temporal_localization_save_dir_name); os.makedirs(temporal_localization_save_dir, exist_ok=True)
tl_converstaion_template = 'tl_video_conversations_all_{}.json'.format
tl_generation_freq = generation_freq

if generate_segment_ref:
    tl_converstaion_template = 'tl_video_conversations_all_wRef_{}.json'.format
else:
    tl_generation_freq = tl_generation_freq * 2

# %% [markdown]
# # Generate the conversations of accident recognition and description.

# %%
rd_question_templates = [
    "[RD] Does this video capture a traffic accident?",
    "[RD] Is a traffic accident occurring at any point in this video?",
    "[RD] Can you detect any traffic collisions in this video?",
    "[RD] Is there evidence of a road traffic accident visible in this video clip?",
    "[RD] Throughout this video, is there an incident involving a traffic accident?",
    "[RD] Do you observe a car accident happening in the sequence of this video?",
    "[RD] Does this video document any vehicular collisions or crashes?",
    "[RD] Can you point out if there's a traffic-related accident depicted in this video?",
    "[RD] Is there any part of this video that shows a traffic mishap or collision?",
    "[RD] Watch this video and confirm if a traffic accident takes place at any moment.",
]

rd_format_claim = "Please respond with a 'Yes' or 'No'. Following that, describe the video in one sentence."

def recognition_description_conversation_generation(annotation_data, idx):
    question = random.choice(rd_question_templates)
    question += f" {rd_format_claim}"
    question += "\n<video>"
    answer = "No." if annotation_data['accident_type'] == 'normal' else "Yes."
    answer += f" {annotation_data['accident_caption']}"

    conversations = [
            {
                "from": "human",
                "value": question
            },
            {
                "from": "gpt",
                "value": answer
            }            
        ]

    output_structure = {
        "id": idx,  
        "video": annotation_data["videoPath"],
        "accident_type": annotation_data["accident_type"],
        "videoHeight": annotation_data["videoHeight"],
        "videoWidth": annotation_data["videoWidth"],
        "conversations": conversations
    }

    return output_structure


def generate_conversations(anno_path, output_dir, split):
    annos = json.load(open(anno_path, 'r'))
    outputs = []
    freq = rd_generation_freq if split == 'train' else 1
    for idx, annotation_data in enumerate(annos):
        for i in range(freq):
            output = recognition_description_conversation_generation(annotation_data, idx)
            outputs.append(output)
    output_path = os.path.join(output_dir, rd_converstaion_template(split))
    json.dump(outputs, open(output_path, 'w'), indent=4)
    print(f"Generated {len(outputs)} conversations saved to {output_path}")

input_dir = input_dir
output_dir = recognition_description_save_dir
for split in ["train", "val", "test"]:
    anno_path = f'{input_dir}/{annotation_template(split)}'
    generate_conversations(anno_path, output_dir, split)

json_names = [x for x in os.listdir(output_dir) if x.endswith('.json')]
for json_name in json_names:
    json_path = os.path.join(output_dir, json_name)
    with open(json_path, 'r') as f:
        annos = json.load(f)
    missing_files = []
    for anno in annos:
        print
        if not os.path.exists(os.path.join(video_path, anno['video'])):
            print(anno['video'])
            missing_files.append(anno['video'])
            annos.remove(anno)
    print(f"Missing files: {len(missing_files)}")
    with open(json_path, 'w') as f:
        json.dump(annos, f, indent=4, ensure_ascii=False)


# %% [markdown]
# # Generate the conversations of accident temporal localization.

# %%
tl_question_templates = [
    "[TL] Do you know the exact times the {} kicked off and wrapped up?".format, 
    "[TL] Can you give me the start and end times of the {} in the video?".format, 
    "[TL] Any idea about the start and end time of that {} we saw?".format, 
    "[TL] Show me when the {} gets going and when it's all over?".format, 
    "[TL] What is the start and end time of the {} in the video?".format, 
    "[TL] Could you specify the timing of {}'s onset and conclusion?".format, 
    "[TL] Please specify the precise timing of the {}'s onset and conclusion.".format, 
    "[TL] At what timestamps does the {} commence and finish?".format, 
    "[TL] Can you delineate the duration of the {} from beginning to end?".format, 
    "[TL] When is the {} initiated and terminated in the footage?".format, 
]

tl_answer_templates = [
    "Between {}.".format,
    "In the time period {}.".format,
    "During the span of {}.".format,
    "It happens in {}.".format,
    "At {}.".format,
    "Exactly at {}.".format,
    "Through {}.".format,
    "Within the window of {}.".format,
    "In the {} mark.".format,
    "Around {}.".format,
]

ref_question_templates = [
    "[TL] What's happened during {} in the video?".format,  
    "[TL] What's the incident in the period of {}?".format,  
    "[TL] Maybe something wrong happened during {} in the provided video?".format,  
    "[TL] What's the traffic situation in the period of {}?".format,  
    "[TL] Is the traffic flow captured by the video normal during {}?".format,  
    "[TL] Dose the accident happen during {} in the video?".format,  
    "[TL] Does the video record any traffic disruptions or accidents around {}?".format,  
    "[TL] Is there any indication of an abnormal traffic event during {}?".format,  
    "[TL] Could you identify any mishaps in the time frame of {}?".format,  
    "[TL] Are there signs of vehicular distress or accidents within {}?".format,  
]

ref_answer_templates = [
    "There was a {} that occurred in the observed segment.".format,  
    "A {} is observed, indicating an abnormal situation in the traffic flow.".format,  
    "The footage captures a {} occurring within the captured segment.".format,  
    "There is a disruption in traffic due to a {}.".format,  
    "An unfortunate {} takes place, which is clearly captured in the video.".format,  
    "The segment shows that a {} happened, causing irregular traffic.".format,  
    "A {} was clearly evident at the mentioned timeframe.".format,  
    "The given video segment shows a definite {}.".format,  
    "During the given segment, a {} disrupts the normal flow of traffic.".format,  
    "Within the specified timestamps, a consequential {} is recorded.".format,  
]

tl_format_claim = "The answered timestamp should be formatted in a normalized manner, using {start_time, end_time}."
ref_format_claim = "The timestamp is given by {start_time, end_time} formatted in a normalized manner."

def float2str(x):
    return "{:.2f}".format(float(x))

def temporal_localization_qa_generation(annotation_data, idx, split='train'):
    conversations = []

    question = random.choice(tl_question_templates)("traffic accident")
    question += " " + tl_format_claim
    question += "\n<video>"

    conversations.append({
        "from": "human",
        "value": question
    })
    accident_segment = annotation_data["accident_segments"][0]
    accident_segment = [float2str(x) for x in accident_segment]
    conversations.append({
        "from": "gpt",
        "value": random.choice(tl_answer_templates)("{"+ ", ".join(accident_segment) + "}")
    })
    if generate_object_bbox and annotation_data['accident_objects'] and split == 'train':
        for accident_instance in annotation_data['accident_objects']:
            objects_description_parts = []
            for obj in accident_instance['objects']:
                formatted_bbox = [(f"{b:.3f}") for b in obj['bbox']]
                formatted_bbox = ", ".join(formatted_bbox)
                objects_description_parts.append(f"{obj['label']} located at [{formatted_bbox}]")
            objects_description = " and ".join(objects_description_parts)
            timestamp_bbox_answer = f"At {accident_instance['timestamp']} normalized timestamp, {objects_description} were involved in the accident."
        conversations.append({
            "from": "human",
            "value": "Can you identify the entities, including pedestrians and vehicles, were involved in the traffic accident?"
        })
        conversations.append({
            "from": "gpt",
            "value": timestamp_bbox_answer
        })

    output_structure = {
        "id": idx,  
        "video": annotation_data["videoPath"],
        "accident_type": annotation_data["accident_type"],
        "videoHeight": annotation_data["videoHeight"],
        "videoWidth": annotation_data["videoWidth"],
        "conversations": conversations
    }

    return output_structure

def temporal_reference_qa_generation(annotation_data, idx):
    segment = annotation_data["accident_segments"][0]
    segment = [float2str(x) for x in segment]
    question = random.choice(ref_question_templates)("{"+ ", ".join(segment) + "}")
    question += " " + ref_format_claim
    question += "\n<video>"
    answer = random.choice(ref_answer_templates)("traffic accident")

    conversations = [
            {
                "from": "human",
                "value": question
            },
            {
                "from": "gpt",
                "value": answer
            }            
        ]

    output_structure = {
        "id": idx,  
        "video": annotation_data["videoPath"],
        "accident_type": annotation_data["accident_type"],
        "videoHeight": annotation_data["videoHeight"],
        "videoWidth": annotation_data["videoWidth"],
        "conversations": conversations
    }

    return output_structure


def generate_conversations(anno_path, output_dir, split):
    annos = json.load(open(anno_path, 'r'))
    outputs = []
    freq = tl_generation_freq if split == 'train' else 1
    for idx, annotation_data in enumerate(annos):
        if len(annotation_data["accident_segments"]) > 0 and \
                annotation_data["accident_type"] != 'normal':
            for i in range(freq):
                output = temporal_localization_qa_generation(annotation_data, idx, split)
                outputs.append(output)
            if generate_segment_ref and split == 'train':
                for i in range(freq):
                    output = temporal_reference_qa_generation(annotation_data, idx)
                    outputs.append(output)
    output_path = os.path.join(output_dir, tl_converstaion_template(split))
    json.dump(outputs, open(output_path, 'w'), indent=4)
    print(f"Generated {len(outputs)} conversations saved to {output_path}")

input_dir = input_dir
output_dir = temporal_localization_save_dir
for split in ["train", "val", "test"]:
    anno_path = f'{input_dir}/{annotation_template(split)}'
    generate_conversations(anno_path, output_dir, split)

json_names = [x for x in os.listdir(output_dir) if x.endswith('.json')]
for json_name in json_names:
    json_path = os.path.join(output_dir, json_name)
    with open(json_path, 'r') as f:
        annos = json.load(f)
    missing_files = []
    for anno in annos:
        print
        if not os.path.exists(os.path.join(video_path, anno['video'])):
            print(anno['video'])
            missing_files.append(anno['video'])
            annos.remove(anno)
    print(f"Missing files: {len(missing_files)}")
    with open(json_path, 'w') as f:
        json.dump(annos, f, indent=4, ensure_ascii=False)



