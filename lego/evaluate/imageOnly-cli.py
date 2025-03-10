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
    image_path = args.image_file
    image = load_image_square(image_path,image_processor,image_aspect_ratio='resize')
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(dtype=torch.bfloat16).cuda()
    if args.fixQ and args.task != 'og':
        if 'cc' in args.task:
            inp = f"Please respond with a 'Yes' or 'No' to indicate whether there is an accident occurring in the image. Be very clear and straightforward with your answer. Following that, describe the image in one sentence."
        elif args.task == 'ag':
            inp = f"Please provide the bounding box coordinate of the traffic accident depicted in the image. Use the normalized bounding box format, specified as [x_min, y_min, x_max, y_max]."
    else:
        inp = question.replace('\n<image>', '')

    if args.shuffix == 'addTaskFlag':
        inp = f" {args.task[:2].upper()}" + inp

    # print(f"{roles[1]}: ", end="")
    if image is not None:
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IMAGE_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * CONFIG.image_token_len + DEFAULT_IMAGE_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None

    
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
    parser.add_argument("--image_file", type=str, default=None)
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--task", type=str, default='ag', choices=['ag','og', 'ccI'])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--accidentOnly", action="store_true")
    parser.add_argument("--fixQ", action="store_true")
    parser.add_argument("--shuffix", type=str, default="")
    args = parser.parse_args()
    args.accidentOnly = 'cc' not in args.task
    args.fixQ = True
    if args.accidentOnly:
        print("Eval on Accident Image Only !!!!!!!!!!!!!!!!!!!!!!!!!")
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
        args.image_file = os.path.join(args.image_dir, data[i]['image'])
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
