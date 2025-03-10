import argparse
import torch
import sys
sys.path.append('/workspace/GroundingGPT')
from lego.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_SOUND_TOKEN
from lego.conversation import SeparatorStyle
from lego import conversation as conversation_lib
from lego.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, load_image_square, postprocess_output
from lego.model.builder import CONFIG, load_pretrained_model
from video_llama.processors.video_processor import load_video
from video_llama.models.ImageBind.data import load_and_transform_audio_data
from lego.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN, \
                           DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN, DEFAULT_SOUND_PATCH_TOKEN, DEFAULT_SOUND_START_TOKEN, DEFAULT_SOUND_END_TOKEN

from tqdm import tqdm

def single_process(args, model_bag, question):
    model, tokenizer, image_processor, video_transform, context_len = model_bag
    conv = conversation_lib.default_conversation.copy()
    roles = conv.roles
    image_path = None
    image_tensor = None
    video_tensor = None
    sound_tensor = None
    image = None
    # try :
    video_path = args.video_file
    video = load_video(
            video_path = video_path,
            n_frms = model.config.max_frame,
            height = 224,
            width = 224,
            sampling ="uniform", return_msg = False)
    video_tensor = video_transform(video).unsqueeze(0).to(CONFIG.device, dtype=torch.bfloat16)
    if args.fixQ:
        if 'cc' in args.task:
            inp = f"Please respond with a 'Yes' or 'No' to indicate whether there is an accident captured in the video. Be very clear and straightforward with your answer. Following that, describe the video in one sentence."       
        elif args.task == 'tl':
            inp = "Please specify the start and end timestamps of the traffic accident captured in the video, adhering to the format of normalized timestamps, denoted as {start_time, end_time}."
    else:
        inp = question.replace('\n<video>', '')

    if args.shuffix == 'addTaskFlag':
        inp = f" {args.task[:2].upper()}" + inp
        
    if video is not None:
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * CONFIG.video_token_len + DEFAULT_VIDEO_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_VIDEO_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        video = None

    
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            videos=video_tensor,
            sounds=sound_tensor,
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens, 
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    if image_path is not None:
        outputs = postprocess_output(outputs, image_path)
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    # print(outputs)
    return outputs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ckpt/GroundingGPT-7B")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--task", type=str, default='tl', choices=['tl', 'ccV'])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--accidentOnly", action="store_true")
    parser.add_argument("--fixQ", action="store_true")
    parser.add_argument("--shuffix", type=str, default="")
    args = parser.parse_args()
    args.accidentOnly = 'cc' not in args.task
    args.fixQ = True
    if args.accidentOnly:
        print("Eval on Accident Videos Only !!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print("Only classify the Accident or Normal !!!!!!!!!!!!!!!!!")
    model_bag = load_pretrained_model(args.model_path, args.model_base)

    def get_question(item):
        conversation = item['conversations']
        question = conversation[0]['value']
        return question

    import os
    import json 
    args.output_dir = os.path.join(args.output_dir, os.path.basename(args.model_path.strip('/')), args.task)
    print(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.json_path) as f:
        data = json.load(f)
    output_data = []
    for i in tqdm(range(len(data))):
        args.video_file = os.path.join(args.video_dir, data[i]['video'])
        question = get_question(data[i])
        if args.accidentOnly and 'no' in data[i]['conversations'][1]['value'].lower():
            continue
        else:
            response = single_process(args, model_bag, question)
            data[i]['response'] = response
            output_data.append(data[i])
    output_file = os.path.basename(args.json_path).replace('.json', f'_{args.shuffix}_response.json')
    output_path = os.path.join(args.output_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
