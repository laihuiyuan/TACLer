import os
import json
import argparse
import random
import torch
from math import comb
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import vllm.envs as envs
from vllm import LLM, SamplingParams

from utils import extract_answer, grade_answer_sympy, grade_answer_mathd

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

def parse_list(arg):
    return arg.split(',')


def eval_result(out_file, pass_k, sample_n):

    correct_cnt = 0
    pass_at_k_list = []
    with open(out_file, 'r') as f:
        for example in f.readlines():
            example = json.loads(example)
            is_correct_list = example['list_correct']
            is_correct = any(is_correct_list)
            if is_correct:
                correct_cnt += 1

            correct_answers = sum(is_correct_list)
            if correct_answers > 0:
                if sample_n - correct_answers < pass_k:
                    pass_at_k = 1
                else:
                    pass_at_k = 1 - (comb(sample_n - correct_answers, pass_k) / comb(sample_n, pass_k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0)

    average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
    print(f"Pass@{pass_k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
    print(f"Pass@{sample_n}: {correct_cnt}/{len(pass_at_k_list)} = {correct_cnt / len(pass_at_k_list):.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--think_mode', action='store_true', help='think mode')
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument('--sample_n', default=16, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument('--stop', type=str, nargs='+', default=['\n</solution>'])
    parser.add_argument("--pass_k", default=1, type=int, help="pass@k")
    parser.add_argument("--max_tokens", default=32768, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    # top_p must be 1 when using greedy
    args.top_p = 1 if args.temperature == 0 else args.top_p  

    print(f"current stop list: {args.stop}")
    return args


def infer(args):
    print(f"current eval model: {args.model_path}")

    model = args.model_path.strip("/").split("/")[-1]
    data_name = args.data_path.split("/")[-1].split(".")[0]
    if args.think_mode:
        out_file = f'{args.output_dir}/{model}/{data_name}_k{args.sample_n}_tk.jsonl'
    else:
        out_file = f'{args.output_dir}/{model}/{data_name}_k{args.sample_n}_nt.jsonl'

    if os.path.exists(out_file):
        print(f"{out_file} already exists. Skipping generation and proceeding to evaluation.")
        eval_result(out_file, args.pass_k, args.sample_n)
        return
    os.makedirs(f'{args.output_dir}/{model}', exist_ok=True)

    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP = "0.0.0.0" or "127.0.0.1"
    print(f"available_gpus: {available_gpus}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    with open(args.data_path, 'r') as f:
        dataset = json.load(f)

    prompt_batch = []
    for example in tqdm(dataset, total=len(dataset)):
        think_step = " Let's think step by step and output the final answer within \\boxed{}."
        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": example['problem'] + think_step}
        ]
        cur_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if not args.think_mode:
            cur_prompt += 'Okay, I think I can solve it directly.\n</think>\n\n'
        prompt_batch.append(cur_prompt)
    print(prompt_batch[0])

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.sample_n,
        top_p=args.top_p,
        # stop=args.stop,
        )
    llm = LLM(
        seed=args.seed,
        model=args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=len(available_gpus),
        )
    completions = llm.generate(prompt_batch, sampling_params)

    correct_cnt = 0
    file_outputs = []
    for i in range(len(completions)):

        truths = dataset[i]['answer']
        if isinstance(truths, (str, float, int)):
            truths = [truths]

        processed_truths = []
        for truth in truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_truths.append(processed_truth)
            else:
                processed_truths.append(truth)

        responses = [completions[i].outputs[j].text for j in range(len(completions[i].outputs))]
        model_solutions = []
        for response in responses:
            if THOUGHT_DELIMITER_END in response:
                model_solutions.append(response.split(THOUGHT_DELIMITER_END)[1])
            else:
                model_solutions.append(response)
        pred_answers = [extract_answer(solution) for solution in model_solutions]

        file_outputs.append({
            "question": dataset[i]['problem'],
            "responses": responses,
        })

        is_correct_list = []
        for pred in pred_answers:
            if pred is None:
                is_correct_list.append(False)
            else:
                temp = [(grade_answer_mathd(pred, gold) or grade_answer_sympy(pred, gold)) for gold in processed_truths]
                is_correct_list.append(any(temp))

        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1
        file_outputs[i]['pred_answer'] = pred_answers
        file_outputs[i]['gold_answer'] = processed_truths 
        file_outputs[i]['correctness'] = is_correct
        file_outputs[i]['list_correct'] = is_correct_list

    with open(out_file, 'w', encoding='utf-8') as f:
        count = 0
        for d in tqdm(file_outputs, "Writing results: "):
            f.write(json.dumps(d, ensure_ascii=False))
            f.write("\n")
            count += 1
            if count % 10 == 0:
                f.flush()
        f.flush()

    eval_result(out_file, args.pass_k, args.sample_n)


if __name__ == "__main__":
    args = parse_args()
    infer(args)
