import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from mova.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, ROUTING_PROMPT
from mova.conversation import conv_templates, SeparatorStyle
from mova.model.builder import load_pretrained_model
from mova.utils import disable_torch_init
from mova.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math
import copy 


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


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            routing_qs = copy.deepcopy(ROUTING_PROMPT) + qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                routing_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + routing_qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                routing_qs = DEFAULT_IMAGE_TOKEN + '\n' + routing_qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

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

            if isinstance(image_processor, list):
                image_tensor_0 = process_images([image], image_processor[0], model.config)[0]
                image_tensor_1 = process_images([image], image_processor[1], model.config)[0]
                image_tensor = torch.cat((image_tensor_0, image_tensor_1), dim=0).unsqueeze(0).bfloat16().cuda()
                high_image_tensor = process_images([image], image_processor[2], model.config)[0].unsqueeze(0).bfloat16().cuda()
                flattened_image_tensor = process_images([image], image_processor[3], model.config)[0].unsqueeze(0).bfloat16().cuda()
            else:
                image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).bfloat16().cuda()
                high_image_tensor = image_tensor
                flattened_image_tensor = image_tensor

            # prompts = [["question 0: <q>\n\n".replace("<q>", qs.replace("<image>\n", "").lower())]]
            prompts = [[qs.replace("<image>\n", "").lower()]]
            routing_weight_tensor = torch.Tensor([0]*7).unsqueeze(0).bfloat16().cuda()
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
            routing_weight_tensor = get_routing_weights(outputs).bfloat16().cuda()

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

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    eval_model(args)
