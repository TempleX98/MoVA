import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from mova.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, ROUTING_PROMPT
from mova.conversation import conv_templates, SeparatorStyle
from mova.model.builder import load_pretrained_model
from mova.utils import disable_torch_init
from mova.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import copy


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_routing_weights(response):
    response = ','.join(response.split(',')[:3])    
    result = [0, 0, 0, 0, 0, 0, 0]
    if "A" in response:
        result[0] = 1
    if "B" in response:
        result[1] = 1
    if "C" in response:
        result[2] = 1
    if "D" in response:
        result[3] = 1
    if "E" in response:
        result[4] = 1
    if "F" in response:
        result[5] = 1
    if "G" in response:
        result[6] = 1
    return torch.Tensor(result).unsqueeze(0)


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    with open(args.questions_file) as f:
        llvqa_data = json.load(f)

    for i, llddata in enumerate(tqdm(llvqa_data)):
        image_file = llddata["img_path"]
        if args.lang == "en":
            message = llddata["question"] + \
                "\nChoose between one of the options as follows:\n"
        elif args.lang == "zh":
            message = llddata["question"] + "\在下列选项中选择一个:\n"
        else:
            raise NotImplementedError(
                "Q-Bench does not support languages other than English (en) and Chinese (zh) yet. Contact us (https://github.com/VQAssessment/Q-Bench/) to convert  Q-Bench into more languages.")
        for choice, ans in zip(["A.", "B.", "C.", "D."], llddata["candidates"]):
            message += f"{choice} {ans}\n"
        qs = message
        routing_qs = copy.deepcopy(ROUTING_PROMPT) + qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                DEFAULT_IM_END_TOKEN + '\n' + qs
            routing_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                DEFAULT_IM_END_TOKEN + '\n' + routing_qs                
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            routing_qs = DEFAULT_IMAGE_TOKEN + '\n' + routing_qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], routing_qs)
        conv.append_message(conv.roles[1], None)
        routing_qs = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        routing_input_ids = tokenizer_image_token(routing_qs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')

        if isinstance(image_processor, list):
            image_tensor_0 = process_images(
                [image], image_processor[0], model.config
            )[0]
            image_tensor_1 = process_images(
                [image], image_processor[1], model.config
            )[0]
            image_tensor = torch.cat((image_tensor_0, image_tensor_1), dim=0)
            high_image_tensor = process_images(
                [image], image_processor[2], model.config
            )[0]
            flattened_image_tensor = process_images(
                [image], image_processor[3], model.config
            )[0]
        else:
            image_tensor = process_images(
                [image], image_processor, model.config
            )[0]
            high_image_tensor = image_tensor
            flattened_image_tensor = image_tensor

        prompts = [[qs.replace("<image>\n", "").lower()]]
        image_tensor = image_tensor.cuda().bfloat16().unsqueeze(0)
        high_image_tensor = high_image_tensor.cuda().bfloat16().unsqueeze(0)
        flattened_image_tensor = flattened_image_tensor.cuda().bfloat16().unsqueeze(0)
        routing_weight_tensor = torch.Tensor([0]*7).cuda().bfloat16().unsqueeze(0)

        # Obtain base vision feature
        cached_features = model.update_cached_features(
            image_tensor, 
            high_image_tensor, 
            flattened_image_tensor, 
            routing_weights=routing_weight_tensor)
        with torch.inference_mode():
            output_ids = model.generate(
                routing_input_ids,
                images=image_tensor,
                high_images=high_image_tensor,
                flattened_patches=flattened_image_tensor,
                routing_weights=routing_weight_tensor,
                cached_features=cached_features,
                prompts=prompts,                
                has_routing=[True],
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=16,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        routing_weight_tensor = get_routing_weights(outputs).cuda().bfloat16()

        # Update vision feature
        cached_features = model.update_cached_features(
            image_tensor, 
            high_image_tensor, 
            flattened_image_tensor, 
            routing_weights=routing_weight_tensor,
            cached_features=cached_features)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                high_images=high_image_tensor,
                flattened_patches=flattened_image_tensor,
                routing_weights=routing_weight_tensor,
                cached_features=cached_features,
                prompts=prompts,                
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        llddata["response"] = outputs
        with open(args.answers_file, "a") as wf:
            json.dump(llddata, wf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--regen", action="store_true", default=False)
    args = parser.parse_args()
         
    if os.path.exists(args.answers_file) and not args.regen:
        print("{} already exists, won't regen again.".format(args.answers_file))
    else:
        eval_model(args)
