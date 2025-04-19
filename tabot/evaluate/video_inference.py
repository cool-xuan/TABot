import argparse
import torch
import os
import json
from tqdm import tqdm
from tabot.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_VIDEO_TOKEN,
    DEFAULT_VIDEO_PATCH_TOKEN,
    DEFAULT_VIDEO_START_TOKEN,
    DEFAULT_VIDEO_END_TOKEN
)
from tabot.conversation import SeparatorStyle
from tabot import conversation as conversation_lib
from tabot.mm_utils import (
    tokenizer_image_token,
    KeywordsStoppingCriteria
)
from tabot.model.builder import CONFIG, load_pretrained_model
from video_llama.processors.video_processor import load_video


def generate_response(args, model_bundle, question):
    """
    Process a single sample through the video-based multimodal model.
    """
    model, tokenizer, _, video_transform, _ = model_bundle

    # Load and preprocess video
    video_path = args.video_file
    video = load_video(
        video_path=video_path,
        n_frms=model.config.max_frame,
        height=224,
        width=224,
        sampling="uniform",
        return_msg=False
    )
    video_tensor = video_transform(video).unsqueeze(0).to(CONFIG.device, dtype=torch.bfloat16)

    
    prompt_text = question.replace('\n<video>', '')
   # Prepare multimodal input with video token(s)
    if model.config.mm_use_im_start_end:
        prompt_text = (
            DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * CONFIG.video_token_len +
            DEFAULT_VIDEO_END_TOKEN + '\n' + prompt_text
        )
    else:
        prompt_text = DEFAULT_VIDEO_TOKEN + '\n' + prompt_text

    # Build conversation context
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize input
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).cuda()

    # Setup stopping criteria
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Generate model output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            videos=video_tensor,
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    # Decode and postprocess output
    generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    if generated_text.endswith(stop_str):
        generated_text = generated_text[:-len(stop_str)]

    return generated_text


def get_first_question(conversations):
    """
    Extract the first question from the conversation list.
    """
    return conversations[0]['value']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ckpt/GroundingGPT-7B")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    # Load pretrained model
    model_bundle = load_pretrained_model(args.model_path)

    # Create output directory
    output_folder = args.output_dir
    os.makedirs(output_folder, exist_ok=True)
    print(f"[INFO] Output directory: {output_folder}")

    # Load input JSON data
    with open(args.json_path) as f:
        input_data = json.load(f)

    output_data = []
    for item in tqdm(input_data, desc="Processing samples"):
        args.video_file = os.path.join(args.video_dir, item['video'])
        question = item['conversations'][0]['value']
        response = generate_response(args, model_bundle, question)

        # Add label and prediction to output
        item['label'] = item['conversations'][1]['value']
        item['predict'] = response
        output_data.append(item)

    # Save outputs
    output_filename = os.path.basename(args.json_path)
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"[INFO] Output written to: {output_path}")


if __name__ == "__main__":
    main()
