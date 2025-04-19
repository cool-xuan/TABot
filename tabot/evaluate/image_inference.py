import argparse
import torch
import os
import json
from tqdm import tqdm
from tabot.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_START_TOKEN,
    DEFAULT_IMAGE_END_TOKEN
)
from tabot.conversation import SeparatorStyle
from tabot import conversation as conversation_lib
from tabot.mm_utils import (
    tokenizer_image_token,
    KeywordsStoppingCriteria,
    load_image_square,
    postprocess_output
)
from tabot.model.builder import CONFIG, load_pretrained_model


def generate_response(args, model_bundle, question):
    """
    Process a single sample through the multimodal model.
    """
    model, tokenizer, image_processor, _, _ = model_bundle

    # Load and preprocess image
    image_path = args.image_file
    image = load_image_square(image_path, image_processor, image_aspect_ratio='resize')
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    image_tensor = image_tensor.to(dtype=torch.bfloat16).cuda()
    
    # Clean the question prompt
    prompt_text = question.replace('\n<image>', '')
    
    # Prepare multimodal input with image token(s)
    if model.config.mm_use_im_start_end:
        prompt_text = (
            DEFAULT_IMAGE_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * CONFIG.image_token_len +
            DEFAULT_IMAGE_END_TOKEN + '\n' + prompt_text
        )
    else:
        prompt_text = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text

    # Build conversation context
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize the input
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).cuda()

    # Setup stopping criteria
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Generate model output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
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
    if image_path is not None:
        generated_text = postprocess_output(generated_text, image_path)

    return generated_text


def get_first_question(conversations):
    """
    Extract the first question from the conversation list.
    """
    return conversations[0]['value']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ckpt/TABot-7B")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
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
        args.image_file = os.path.join(args.image_dir, item['image'])
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
