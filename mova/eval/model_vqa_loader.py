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
from torch.utils.data import Dataset, DataLoader

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

    
# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        routing_qs = copy.deepcopy(ROUTING_PROMPT) + qs
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            routing_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + routing_qs
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

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

        if isinstance(self.image_processor, list):
            image_tensor_0 = process_images(
                [image], self.image_processor[0], self.model_config
            )[0]
            image_tensor_1 = process_images(
                [image], self.image_processor[1], self.model_config
            )[0]
            image_tensor = torch.cat((image_tensor_0, image_tensor_1), dim=0)
            high_image_tensor = process_images(
                [image], self.image_processor[2], self.model_config
            )[0]
            flattened_image_tensor = process_images(
                [image], self.image_processor[3], self.model_config
            )[0]
        else:
            image_tensor = process_images(
                [image], self.image_processor, self.model_config
            )[0]
            high_image_tensor = image_tensor
            flattened_image_tensor = image_tensor

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        routing_input_ids = tokenizer_image_token(routing_qs, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, routing_input_ids, image_tensor, high_image_tensor, flattened_image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, routing_input_ids, image_tensors, high_image_tensors, flattened_image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    routing_input_ids = torch.stack(routing_input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    high_image_tensors = torch.stack(high_image_tensors, dim=0)
    flattened_image_tensors = torch.stack(flattened_image_tensors, dim=0)
    return input_ids, routing_input_ids, image_tensors, high_image_tensors, flattened_image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.eval()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, routing_input_ids, image_tensor, high_image_tensor, flattened_image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        prompts = [[cur_prompt.replace("<image>\n", "").lower()]]

        image_tensor = image_tensor.to(dtype=torch.bfloat16, device="cuda", non_blocking=True)
        high_image_tensor = high_image_tensor.to(dtype=torch.bfloat16, device="cuda", non_blocking=True)
        flattened_image_tensor = flattened_image_tensor.to(dtype=torch.bfloat16, device="cuda", non_blocking=True)
        routing_weight_tensor = torch.Tensor([0]*7).to(dtype=torch.bfloat16, device='cuda', non_blocking=True).unsqueeze(0)

        # Obtain base vision feature
        cached_features = model.update_cached_features(
            image_tensor, 
            high_image_tensor, 
            flattened_image_tensor, 
            routing_weights=routing_weight_tensor)
        routing_input_ids = routing_input_ids.to(device='cuda', non_blocking=True)
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
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=16,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        routing_weight_tensor = get_routing_weights(outputs).to(device='cuda', non_blocking=True)

        # Update vision feature
        cached_features = model.update_cached_features(
            image_tensor, 
            high_image_tensor, 
            flattened_image_tensor, 
            routing_weights=routing_weight_tensor,
            cached_features=cached_features)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                high_images=high_image_tensor,
                flattened_patches=flattened_image_tensor,
                routing_weights=routing_weight_tensor,
                cached_features=cached_features,
                prompts=prompts,                
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
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
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
