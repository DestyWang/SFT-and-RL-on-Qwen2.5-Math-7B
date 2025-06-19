import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoModel, Trainer, TrainingArguments, HfArgumentParser,
    AutoModelForSequenceClassification
)
import trl
from datasets import load_dataset, Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, RewardTrainer, RewardConfig
from peft import LoraConfig, get_peft_model, TaskType
import pickle
import json
import datetime
now = datetime.datetime.now()
print("当前时间:", now.hour, "时", now.minute, "分", now.second, "秒")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_NAME = "/home/bcl/wanghongyu/other/Qwen/Qwen2.5-Math-7B-Instruct"
MODEL_NAME = "/home/bcl/wanghongyu/other/Qwen/SFT/output/checkpoint-900"
output_dir = "/home/bcl/wanghongyu/other/Qwen/RL/output1"
reward_model_dir = os.path.join(output_dir, "reward_model_6_15_17_12")
if not os.path.exists(reward_model_dir):
    os.makedirs(reward_model_dir, exist_ok=True)
data_used_dir = "/home/bcl/wanghongyu/other/Qwen/RL/data/reward_data"
# data_used_dir = os.path.join(output_dir, "data")
# if not os.path.exists(data_used_dir):
#     os.makedirs(reward_model_dir, exist_ok=True)
# train_data_size = 90000

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

def preprocess_function_reward(examples):
    """
    预处理函数，用于奖励模型训练
    将数据转换为chosen/rejected格式，输出格式类似trl-lib/ultrafeedback_binarized
    """
    chosen = []
    rejected = []

    system_prompt = ("You are an AI assistant skilled in mathematical reasoning. "
                     "Please solve the problem using concise step-by-step reasoning process. "
                     "Put your final answer within \\boxed{}.")

    # 处理每个样本
    for problem, generations, correctness_list in zip(
            examples['problem'],
            examples['generations'],
            examples['correctness_math_verify']
    ):
        # 分离正确和错误的生成结果
        correct_generations = []
        incorrect_generations = []

        # 确保correctness_list是列表格式
        if not isinstance(correctness_list, list):
            correctness_list = [correctness_list]

        # 确保generations是列表格式
        if not isinstance(generations, list):
            generations = [generations]

        for gen, is_correct in zip(generations, correctness_list):
            if is_correct:
                correct_generations.append(gen)
            else:
                incorrect_generations.append(gen)

        # 如果同时有正确和错误的答案，创建对比对
        if correct_generations and incorrect_generations:
            # 随机选择一个正确答案作为chosen
            chosen_response = random.choice(correct_generations)
            # 随机选择一个错误答案作为rejected
            rejected_response = random.choice(incorrect_generations)

            # 构建chosen对话格式
            chosen_conversation = [
                {
                    "content": system_prompt,
                    "role": "system"
                },
                {
                    "content": f"Problem: {problem}",
                    "role": "user"
                },
                {
                    "content": chosen_response,
                    "role": "assistant"
                }
            ]

            # 构建rejected对话格式
            rejected_conversation = [
                {
                    "content": system_prompt,
                    "role": "system"
                },
                {
                    "content": f"Problem: {problem}",
                    "role": "user"
                },
                {
                    "content": rejected_response,
                    "role": "assistant"
                }
            ]

            chosen.append(chosen_conversation)
            rejected.append(rejected_conversation)

    print(f"Generated {len(chosen)} chosen/rejected pairs from {len(examples['problem'])} examples")

    return {
        "chosen": chosen,
        "rejected": rejected
    }

def preprocess_function_reward_cut(examples, max_tokens=1024, tokenizer=tokenizer):
    """
    预处理函数，用于奖励模型训练。
    - 筛选出每个样本中最短的正确答案和最短的错误答案。
    - 超过 max_tokens 的回答将被跳过。
    - 返回的格式为 trl 格式：{"chosen": [...], "rejected": [...]}
    """

    assert tokenizer is not None, "tokenizer 不能为空，用于计算 token 数。"

    chosen = []
    rejected = []

    system_prompt = ("You are an AI assistant skilled in mathematical reasoning. "
                     "Please solve the problem using concise step-by-step reasoning process. "
                     "Put your final answer within \\boxed{}.")

    for problem, generations, correctness_list in zip(
            examples['problem'],
            examples['generations'],
            examples['correctness_math_verify']
    ):
        # 转换为列表以防止是单个字符串
        generations = generations if isinstance(generations, list) else [generations]
        correctness_list = correctness_list if isinstance(correctness_list, list) else [correctness_list]

        # 分别存储正确和错误的回答
        correct_generations = [gen for gen, correct in zip(generations, correctness_list) if correct]
        incorrect_generations = [gen for gen, correct in zip(generations, correctness_list) if not correct]

        if correct_generations and incorrect_generations:
            # 选取最短的正确/错误答案
            chosen_response = min(correct_generations, key=lambda x: len(x))
            rejected_response = min(incorrect_generations, key=lambda x: len(x))

            # 构建完整对话文本用于tokenizer
            def build_conv(response):
                return f"{system_prompt}\nProblem: {problem}\n{response}"

            chosen_text = build_conv(chosen_response)
            rejected_text = build_conv(rejected_response)

            # 判断 token 数是否超出阈值
            if len(tokenizer(chosen_text, truncation=False)["input_ids"]) > max_tokens or \
               len(tokenizer(rejected_text, truncation=False)["input_ids"]) > max_tokens:
                continue

            # 构建 chosen/rejected 格式
            chosen_conversation = [
                {"content": system_prompt, "role": "system"},
                {"content": f"Problem: {problem}", "role": "user"},
                {"content": chosen_response, "role": "assistant"}
            ]

            rejected_conversation = [
                {"content": system_prompt, "role": "system"},
                {"content": f"Problem: {problem}", "role": "user"},
                {"content": rejected_response, "role": "assistant"}
            ]

            chosen.append(chosen_conversation)
            rejected.append(rejected_conversation)

    print(f"Generated {len(chosen)} chosen/rejected pairs from {len(examples['problem'])} examples")
    return {"chosen": chosen, "rejected": rejected}

def load_reward_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    读取生成的reward数据集

    Args:
        file_path: 数据文件路径

    Returns:
        数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

print("Loading data...")

# # [:{train_data_size}]
# dataset = load_dataset("open-r1/OpenR1-Math-220k", split=f"train")
# # reward_dataset = dataset.map(preprocess_function_reward, batched=True, remove_columns=dataset.column_names)
# dataset = preprocess_function_reward_cut(dataset)
# dataset = Dataset.from_dict(dataset)
# with open(os.path.join(data_used_dir,"data_used.pkl"), "wb") as file:
#     pickle.dump(dataset, file)

# with open(os.path.join(data_used_dir,"data_used.pkl"), "rb") as file:
#     dataset = pickle.load(file)

dataset = load_reward_dataset(os.path.join(data_used_dir, "reward_data_all.json"))
dataset = Dataset.from_list(dataset)


print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                           # torch_dtype=torch.float16, #不能直接用，还需要修改其他地方，否则loss会变成nan
                                                           num_labels=1,  # 回归任务，输出单个奖励分数
                                                           trust_remote_code=True,
                                                           # use_cache=False,
                                                           pad_token_id=tokenizer.pad_token_id,
                                                           )
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type=TaskType.SEQ_CLS,  # 序列分类任务
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    init_lora_weights=True,
)

reward_config = RewardConfig(
    output_dir=reward_model_dir,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    max_grad_norm=1,
    weight_decay=0.0,
    logging_steps=2,
    save_steps=50,
    save_total_limit=10,
    report_to=None,  # 不使用wandb等追踪工具
    remove_unused_columns=True,
    dataloader_drop_last=True,
    max_length=1024,
    gradient_checkpointing=False,# 暂时设置为false
    label_names=[],
    # pad_token_id=tokenizer.pad_token_id,
)

trainer = RewardTrainer(
    model=model,
    args=reward_config,
    processing_class=tokenizer,
    train_dataset=dataset,
    peft_config=lora_config,
)

trainer.train()

# 保存最终模型
trainer.save_model(reward_model_dir)
tokenizer.save_pretrained(reward_model_dir)

print(f"Reward model and tokenizer saved to {reward_model_dir}")

# CUDA_VISIBLE_DEVICES=7 nohup python3 -u reward.py > output1/reward.log 2>&1 &
# 963254

