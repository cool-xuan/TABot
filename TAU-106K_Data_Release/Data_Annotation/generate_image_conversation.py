import os
import json
import numpy as np
import random

random_seed = 9826
random.seed(random_seed)
np.random.seed(random_seed)

input_dir = '/path/to/image_annotations/'
image_path = '/path/to/your/images/'
output_dir = 'tasks_epoch3'
generation_freq = 3
generate_ref_question=True

annotation_template   = 'image_annotations_all_{}.json'.format

recognition_description_save_dir_name = 'recognition_description_image'
recognition_description_save_dir = os.path.join(output_dir, recognition_description_save_dir_name); os.makedirs(recognition_description_save_dir, exist_ok=True)
rd_converstaion_template = 'rd_image_conversations_all_{}.json'.format
rd_generation_freq = generation_freq

accident_grounding_dir_name = 'accident_grounding'
accident_grounding_dir = os.path.join(output_dir, accident_grounding_dir_name); os.makedirs(accident_grounding_dir, exist_ok=True)
ag_converstaion_template = 'ag_image_conversations_all_{}.json'.format
ag_generation_freq = generation_freq

object_grounding_dir_name = 'object_grounding'
object_grounding_dir = os.path.join(output_dir, object_grounding_dir_name); os.makedirs(object_grounding_dir, exist_ok=True)
og_converstaion_template = 'og_image_conversations_all_{}.json'.format
og_generation_freq = generation_freq

if generate_ref_question:
    ag_converstaion_template = 'ag_image_conversations_all_wRef_{}.json'.format
    og_converstaion_template = 'og_image_conversations_all_wRef_{}.json'.format
else:
    ag_generation_freq = ag_generation_freq * 2
    og_generation_freq = og_generation_freq * 2

# %% [markdown]
# # Generate the conversations of accident recognition and description.

# %%
rd_question_templates = [
    "[RD] Does this image depict a traffic accident?",
    "[RD] Is there a traffic collision shown in this picture?",
    "[RD] Can you identify a vehicular accident in this photo?",
    "[RD] Is this image representative of a traffic mishap?",
    "[RD] Does this photograph show any signs of a car crash?",
    "[RD] Are there any indications of a traffic accident in this image?",
    "[RD] Does the scene in this picture involve a traffic accident?",
    "[RD] Can you confirm a vehicle accident occurrence in this picture?",
    "[RD] Is there evidence of a road traffic accident in this image?",
    "[RD] Do you notice any traffic accident scenarios in this picture?",
]

rd_format_claim = "Please respond with a 'Yes' or 'No'. Following that, describe the image in one sentence."

def recognition_description_conversation_generation(annotation_data, idx):
    question = random.choice(rd_question_templates)
    question += f" {rd_format_claim}"
    question += "\n<image>"
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
        "image": annotation_data["imagePath"],
        "accident_type": annotation_data["accident_type"],
        "imageHeight": annotation_data["imageHeight"],
        "imageWidth": annotation_data["imageWidth"],
        "conversations": conversations
    }

    return output_structure

def generate_conversations(anno_path, output_dir, split):
    annos = json.load(open(anno_path, 'r'))
    outputs = []
    for idx, annotation_data in enumerate(annos):
        if split == "train":
            for i in range(rd_generation_freq):
                output = recognition_description_conversation_generation(annotation_data, idx)
                outputs.append(output)
        else:
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
        if not os.path.exists(os.path.join(image_path, anno['image'])):
            print(anno['image'])
            missing_files.append(anno['image'])
            annos.remove(anno)
    print(f"Missing files: {len(missing_files)}")
    with open(json_path, 'w') as f:
        json.dump(annos, f, indent=4, ensure_ascii=False)


# %% [markdown]
# # Generate the spatial grounding conversations for the whole accident.

# %%
bbox_question_templates = [
    "[SG] Where is the {}?".format,
    "[SG] Where is the {} in the image?".format,
    "[SG] Provide the coordinates of the {} in the image?".format,
    "[SG] Can you point out the {} in the image and provide the coordinates of its location?".format,
    "[SG] Help me to locate the {} in the image and give me its coordinates, please.".format,
    "[SG] In the given image, could you find and tell me the coordinates of the {}?".format,
    "[SG] Guide me to the location of the {} within the image by providing its coordinates.".format,
    "[SG] I'd like to know the exact coordinates of the {} in the photo.".format,
    "[SG] Would you kindly provide the coordinates of the {} located in the picture?".format,
    "[SG] Can you find the {} in the image and give me the coordinates of where it is located?".format,
]

ref_question_templates = [
    "[SG] What's happened in the region of {}?".format, 
    "[SG] What's the incident in the area of {}?".format, 
    "[SG] Can you describe the event in the region of {}?".format, 
    "[SG] Please provide details of the incident in the area of {}?".format, 
    "[SG] Describe the incident in the region of {}?".format, 
    "[SG] Is there any evidence of a traffic accident within {}?".format,  
    "[SG] What events are depicted in the specified region {}?".format,  
    "[SG] Is there a record of a traffic disruption within the boundaries of {}?".format,  
    "[SG] What anomalies can be identified in the designated area of {}?".format,  
    "[SG] Could you report any collisions detected in the area demarcated by {}?".format,  
]

ref_answer_templates = [
    "The {} is happened at the grounded area".format,
    "The {} took place within the given coordinates".format,
    "The {} can be found in the given region".format,
    "The {} is reported to have happened within the area".format,
    "A {} is reported within the area".format,
    "Within the specified region, a {} has occurred".format,
    "The area contains signs of a {}".format,
    "There appears to be a {} in the location".format,
    "An {} is observable".format,
    "The captured image data confirms the occurrence of a {}".format,
]


bbox_format_claim = "The answer should be given in normalized [x_min, y_min, x_max, y_max] format."
ref_format_claim = "The cooridinates is given in normalized [x_min, y_min, x_max, y_max] format."

def format_bbox(bbox):
    return [round(x, 3) for x in bbox]
def format_bbox_3f(bbox):
    bbox = [f"{x:.3f}" for x in bbox]
    return "[" + ", ".join(bbox) + "]"

def accident_grounding_qa_generation(annotation_data, idx, conv_type='bbox'):
    if annotation_data["accident_objects"]:
        bboxes = [obj["bbox"] for obj in annotation_data["accident_objects"]]
        min_x = min(bbox[0] for bbox in bboxes)
        min_y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[2] for bbox in bboxes)
        max_y = max(bbox[3] for bbox in bboxes)
        accident_bbox = [min_x, min_y, max_x, max_y]
        accident_bbox = format_bbox(accident_bbox)
    else:
        accident_bbox = None

    def bbox_conversation(accident_bbox):
        question = random.choice(bbox_question_templates)("traffic accident")
        question += bbox_format_claim
        question += "\n<image>"
        conversations= [{
            "from": "human",
            "value": question
            },
            {
                "from": "gpt",
                "value": f"{format_bbox_3f(accident_bbox)}."
            }]
        return conversations

    def ref_conversation(accident_bbox):
        question = random.choice(ref_question_templates)(f"{format_bbox_3f(accident_bbox)}")
        question += ref_format_claim
        question += "\n<image>"
        conversations = [{
            "from": "human",
            "value": question
        },
        {
            "from": "gpt",
            "value": random.choice(ref_answer_templates)("traffic accident"),
        }]

        return conversations

    if conv_type == 'bbox':
        conversations = bbox_conversation(accident_bbox)
    elif conv_type == 'ref':
        conversations = ref_conversation(accident_bbox)
    output_structure = {
        "id": idx,  
        "image": annotation_data["imagePath"],
        "accident_type": annotation_data["accident_type"],
        "imageHeight": annotation_data["imageHeight"],
        "imageWidth": annotation_data["imageWidth"],
        "accident_type": annotation_data["accident_type"],
        "conversations": conversations
    }

    return output_structure

def generate_conversations(anno_path, output_dir, split):
    annos = json.load(open(anno_path, 'r'))
    outputs = []
    for idx, annotation_data in enumerate(annos):
        if annotation_data["accident_objects"]:
            if 'train' in anno_path:
                for i in range(ag_generation_freq):
                    output = accident_grounding_qa_generation(annotation_data, idx, conv_type='bbox')
                    outputs.append(output)
                    if generate_ref_question:
                        output = accident_grounding_qa_generation(annotation_data, idx, conv_type='ref')
                        outputs.append(output)
            else:
                output = accident_grounding_qa_generation(annotation_data, idx, conv_type='bbox')
                outputs.append(output)
    output_path = os.path.join(output_dir, ag_converstaion_template(split))
    json.dump(outputs, open(output_path, 'w'), indent=4, ensure_ascii=False)
    print(f"Generated {len(outputs)} conversations saved to {output_path}")

input_dir = input_dir
output_dir = accident_grounding_dir
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
        if not os.path.exists(os.path.join(image_path, anno['image'])):
            print(anno['image'])
            missing_files.append(anno['image'])
            annos.remove(anno)
    print(f"Missing files: {len(missing_files)}")
    with open(json_path, 'w') as f:
        json.dump(annos, f, indent=4, ensure_ascii=False)


# %% [markdown]
# # Generate the spatial grounding conversations for each object involved in the accident.

# %%
bbox_question_templates = [
    "[SG] Where is the {}?".format,
    "[SG] Where is the {} in the image?".format,
    "[SG] Provide the coordinates of the {} in the image?".format,
    "[SG] Can you point out the {} in the image and provide the coordinates of its location?".format,
    "[SG] Help me to locate the {} in the image and give me its coordinates, please.".format,
    "[SG] In the given image, could you find and tell me the coordinates of the {}?".format,
    "[SG] Guide me to the location of the {} within the image by providing its coordinates.".format,
    "[SG] I'd like to know the exact coordinates of the {} in the photo.".format,
    "[SG] Would you kindly provide the coordinates of the {} located in the picture?".format,
    "[SG] Can you find the {} in the image and give me the coordinates of where it is located?".format,
]

ref_question_templates = [
    "[SG] What's inside the Bbox {}?".format,
    "[SG] Identify the object at {}.".format,
    "[SG] What is in the coordinates {}?".format,
    "[SG] Describe the item within Bbox {}.".format,
    "[SG] What's enclosed in {}?".format,
    "[SG] What does {} contain?".format,
    "[SG] What lies within the Bbox area of {}?".format,
    "[SG] Name the entity located at {}.".format,
    "[SG] What is the object at the coordinates {}?".format,
    "[SG] What can be found at {}?".format,
]

ref_answer_templates = [
    "A {} is detected within the area.".format,
    "The Bbox encloses a {}.".format,
    "Inside the provided coordinates, there is a {}.".format,
    "The object identified is a {}.".format,
    "{} is within the Bbox.".format,
    "Found {} in the specified region.".format,
    "The area contains a {}.".format,
    "{} located at the given coordinates.".format,
    "The object is a {}.".format,
    "A {} is present at the location.".format,
]

accident_involved_templates = [
    "implicated in traffic accidents",
    "involved in road mishaps",
    "associated with collision incidents",
    "linked to traffic collisions",
    "related to roadway accidents",
    "vehicles that have been in accidents",
    "reported in traffic disputes",
    "identified in accident reports",
    "recorded in crash incidents",
    "documented in traffic accident cases",
]


 
ref_format_claim = "The cooridinates is given in normalized [x_min, y_min, x_max, y_max] format."

def format_bbox(bbox):
    return [round(x, 3) for x in bbox]
def format_bbox_3f(bbox):
    bbox = [f"{x:.3f}" for x in bbox]
    return "[" + ", ".join(bbox) + "]"

def object_grounding_qa_generation(annotation_data, obj, idx, conv_type='bbox'):
    object_bbox = obj["bbox"]
    object_bbox = format_bbox(object_bbox)
    object_label = obj["label"]

    def bbox_conversation(object_bbox, object_label):
        question = random.choice(bbox_question_templates)(f"{object_label} {random.choice(accident_involved_templates)}")
        question += bbox_format_claim
        question += "\n<image>"
        conversations= [{
            "from": "human",
            "value": question
            },
            {
                "from": "gpt",
                "value": f"{format_bbox_3f(object_bbox)}."
            }]
        return conversations

    def ref_conversation(object_bbox, object_label):
        question = random.choice(ref_question_templates)(f"{format_bbox_3f(object_bbox)}")
        question += ref_format_claim
        question += "\n<image>"
        conversations = [{
            "from": "human",
            "value": question
        },
        {
            "from": "gpt",
            "value": random.choice(ref_answer_templates)(f"{object_label} {random.choice(accident_involved_templates)}"),
        }]

        return conversations

    if conv_type == 'bbox':
        conversations = bbox_conversation(object_bbox, object_label)
    elif conv_type == 'ref':
        conversations = ref_conversation(object_bbox, object_label)
    output_structure = {
        "id": idx,  
        "image": annotation_data["imagePath"],
        "accident_type": annotation_data["accident_type"],
        "imageHeight": annotation_data["imageHeight"],
        "imageWidth": annotation_data["imageWidth"],
        "accident_type": annotation_data["accident_type"],
        "conversations": conversations
    }

    return output_structure

def generate_conversations(anno_path, output_dir, split):
    annos = json.load(open(anno_path, 'r'))
    outputs = []
    idx_add = 0
    for idx, annotation_data in enumerate(annos):
        objs = annotation_data.get("accident_objects", [])
        if objs:
            for obj in objs:
                if split == 'train':
                    for i in range(og_generation_freq):
                        output = object_grounding_qa_generation(annotation_data, obj, idx+idx_add, conv_type='bbox')
                        outputs.append(output)
                        idx_add += 1
                        if generate_ref_question:
                            output = object_grounding_qa_generation(annotation_data, obj, idx+idx_add, conv_type='ref')
                            outputs.append(output)
                            idx_add += 1
                else:
                    output = object_grounding_qa_generation(annotation_data, obj, idx+idx_add, conv_type='bbox')
                    outputs.append(output)
                    idx_add += 1
    output_path = os.path.join(output_dir, og_converstaion_template(split))
    json.dump(outputs, open(output_path, 'w'), indent=4, ensure_ascii=False)
    print(f"Generated {len(outputs)} conversations saved to {output_path}")

input_dir = input_dir
output_dir = object_grounding_dir
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
        if not os.path.exists(os.path.join(image_path, anno['image'])):
            print(anno['image'])
            missing_files.append(anno['image'])
            annos.remove(anno)
    print(f"Missing files: {len(missing_files)}")
    with open(json_path, 'w') as f:
        json.dump(annos, f, indent=4, ensure_ascii=False)



