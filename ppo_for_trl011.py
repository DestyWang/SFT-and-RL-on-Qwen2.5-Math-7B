import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoModel, Trainer, TrainingArguments,HfArgumentParser,
    AutoModelForSequenceClassification, GenerationConfig
)
import trl
from datasets import load_dataset, Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, RewardTrainer, RewardConfig
from peft import LoraConfig, get_peft_model, TaskType
import datetime
now = datetime.datetime.now()
print("当前时间:", now.hour, "时", now.minute, "分", now.second, "秒")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_NAME = "/home/bcl/wanghongyu/other/Qwen/Qwen2.5-Math-7B-Instruct"
MODEL_NAME = "/home/bcl/wanghongyu/other/Qwen/SFT/output/checkpoint-900"

# reward_model_dir = "/home/bcl/wanghongyu/other/Qwen/RL/final/reward_model/checkpoint-100"
reward_model_dir = "/home/bcl/wanghongyu/other/Qwen/RL/output1/reward_model_6_15_17_12/checkpoint-272"

output_dir = "/home/bcl/wanghongyu/other/Qwen/RL/output1/ppo_model_6_15_17_12"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

train_data_size=1000
system_prompt = ("You are an AI assistant skilled in mathematical reasoning. "
                     "Please solve the problem using concise step-by-step reasoning process. "
                     "Put your final answer within \\boxed{}.")

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function_ppo(examples):
    """
    预处理函数，用于PPO训练
    """
    inputs = []

    system_prompt = ("You are an AI assistant skilled in mathematical reasoning. "
                     "Please solve the problem using concise step-by-step reasoning process. "
                     "Put your final answer within \\boxed{}.")

    for problem in examples['problem']:
        # 格式化问题，加上system prompt
        input_text = f"{system_prompt}\n\nProblem: {problem}\nSolution:"
        inputs.append(input_text)

    return {"query": inputs}

def extract_answer(text):
    """从生成的文本中提取最终答案"""
    # 优先匹配 \boxed{} 中的内容
    box_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    box_match = re.search(box_pattern, text)
    if box_match:
        ans = box_match.group(1).strip()
        return ans if ans else text.strip().split('\n')[-1].strip()

    # 其次匹配常见答案提示
    patterns = [
        r"In summary, the final answer is[:：]?\s*(.*)",
        r"The final answer is[:：]?\s*(.*)",
        r"答案[:：]?\s*(.*)",
        r"The answer is[:：]?\s*(.*)",
        r"The solution is[:：]?\s*(.*)",
        r"The solution to .*? is[:：]?\s*(.*)",
        r"The answer of .*? is[:：]?\s*(.*)",
        r"Answer is[:：]?\s*(.*)",
        r"Answer[:：]?\s*(.*)",
        r"答[:：]?\s*(.*)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            ans = match.group(1).strip()
            # 只取第一行
            return ans.split('\n')[0].strip()

    # 否则取最后一行，去除无关前缀
    last_line = text.strip().split('\n')[-1]
    # 尝试提取数字/表达式
    num_match = re.search(r"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?|[\d\w\(\)\+\-\*/\^=]+)", last_line)
    if num_match:
        return num_match.group(0).strip()
    return last_line.strip()

def check_answer_correctness(generated_answer, correct_answer):
    """检查生成的答案是否正确"""
    if generated_answer is None:
        return False

    try:
        gen_num = float(generated_answer)
        correct_num = float(correct_answer)
        return abs(gen_num - correct_num) < 1e-4
    except (ValueError, TypeError):
        # 字符串比较
        return str(generated_answer).strip().lower() == str(correct_answer).strip().lower()

def calculate_reward(response_text, correct_answer, query_text):
    """
    自定义奖励函数
    """
    # 提取生成的答案
    generated_answer = extract_answer(response_text)

    # 基础奖励：答案正确性
    if check_answer_correctness(generated_answer, correct_answer):
        correctness_reward = 1.0
    else:
        correctness_reward = -1

    # 长度惩罚：避免过长的回答
    length_penalty = max(0, (len(response_text) - 1000) / 1000) * -0.1

    # 格式奖励：是否包含 \boxed{}
    format_reward = 0.2 if "\\boxed{" in response_text else 0.0

    # 步骤奖励：是否包含推理步骤
    step_reward = 0.2 if any(keyword in response_text.lower() for keyword in
                             ["step", "first", "then", "next", "finally", "therefore"]) else 0.0

    total_reward = correctness_reward + length_penalty + format_reward + step_reward
    return total_reward

# 自定义数据收集器
class PPODataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # PPO训练中的数据收集器主要处理query
        if isinstance(batch[0], dict):
            queries = [item['query'] for item in batch]
        else:
            queries = batch

        return queries

print("Loading PPO training dataset...")
# dataset = load_dataset("open-r1/OpenR1-Math-220k", split=f"train[:{train_data_size}]")  # 使用小部分数据进行示例
# dataset = load_dataset("open-r1/OpenR1-Math-220k", split=f"train")
# dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train[10000:11000]")
dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train[20000:21000]")

# 预处理数据
processed_data = preprocess_function_ppo(dataset)
processed_data = Dataset.from_dict(processed_data)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
try:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME,
                                                          reward_adapter=reward_model_dir,
                                                          peft_config=lora_config
                                                          ).to(device)
except:
    pref = "checkpoint-"
    for i in range(1000,100,-50):
        folder = pref + str(i)
        if os.path.exists(os.path.join(reward_model_dir, folder)):
            model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME,
                                                                      reward_adapter=os.path.join(reward_model_dir, folder),
                                                                      peft_config=lora_config
                                                                      ).to(device)
            print(f"Final reward model adapter not found, use {folder} instead.")
            break

# 在创建PPOTrainer之前，添加缺失属性
if not hasattr(model, 'base_model_prefix'):
    # 从pretrained_model获取base_model_prefix
    if hasattr(model, 'pretrained_model'):
        # model.base_model_prefix = model.pretrained_model.base_model_prefix
        model.base_model_prefix = 'pretrained_model'
    else:
        # 通常对于大多数模型都是'model'
        model.base_model_prefix = 'model'

# 确保模型有generation_config属性
if not hasattr(model, 'generation_config'):
    if hasattr(model, 'pretrained_model') and hasattr(model.pretrained_model, 'generation_config'):
        model.generation_config = model.pretrained_model.generation_config
    else:
        model.generation_config = GenerationConfig.from_pretrained(MODEL_NAME)

# 如果是从头开始训练，需要初始化value head
if not hasattr(model, 'v_head') or model.v_head is None:
    print("Initializing value head for PPO training...")

# PPO配置
ppo_config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=1e-5,
    batch_size=2,
    mini_batch_size=1,
    gradient_accumulation_steps=2,
    # early_stopping=True,
    # kl_coef=0.1,
    ppo_epochs=3,
    seed=42
    # stop_token_id=tokenizer.eos_token_id,
)

# 创建数据收集器
data_collator = PPODataCollator(tokenizer)

# 创建PPO训练器
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,
    # reward_model=model,
    # value_model=model,
    tokenizer=tokenizer,
    dataset=processed_data,
    data_collator=data_collator,
)

print("Starting PPO training...")
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 1024,
}

# 训练循环
n_epochs = 2
best_reward = float("-inf")
best_score = float("-inf")
best_loss = float("inf")
for epoch in range(n_epochs):
    print(f"Training: epoch={epoch}.")
    for i in range(0, len(processed_data["query"]), ppo_config.batch_size):
        print(f"Process: {(i + 1) * 100 / len(processed_data['query']):.1f}%.")
        batch_queries = processed_data["query"][i:i + ppo_config.batch_size]
        batch_correct_answers = dataset['answer'][i:i + ppo_config.batch_size]

        # 生成回答
        query_tensors = []
        attention_mask_list = []
        for query in batch_queries:
            query_tensor = tokenizer.encode(query, return_tensors="pt").to(device)
            query_tensors.append(query_tensor.squeeze(0))
        query_tensors = torch.nn.utils.rnn.pad_sequence(query_tensors, batch_first=True, padding_value=tokenizer.pad_token_id)
        query_tensors = query_tensors.to(device)
        # 创建attention mask (1表示真实token，0表示padding)
        attention_mask = (query_tensors != tokenizer.pad_token_id).long()

        # 使用模型生成回答
        # 使用并行要用model->model.module
        response_tensors = ppo_trainer.model.generate(
            query_tensors,
            attention_mask=attention_mask,
            # return_prompt=False,
            **generation_kwargs
        )

        # 计算奖励
        rewards = []
        for j, (query_tensor, response_tensor) in enumerate(zip(query_tensors, response_tensors)):
            # 解码生成的回答
            response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
            query_text = tokenizer.decode(query_tensor, skip_special_tokens=True)

            # 使用reward adapter计算奖励分数
            with torch.no_grad():
                # 构造完整的对话用于奖励模型评分
                # full_text = query_text + response_text
                # full_tensor = tokenizer.encode(full_text, return_tensors="pt").to(device)
                full_text = [
                    {"content": system_prompt, "role": "system"},
                    {"content": f"Problem: {query_text}", "role": "user"},
                    {"content": response_text, "role": "assistant"}
                ]
                full_tensor = tokenizer.apply_chat_template(full_text, return_tensors="pt", tokenize=True).to(device)

                # 使用reward adapter计算分数
                reward_score = ppo_trainer.model.compute_reward_score(
                    input_ids=full_tensor,
                    attention_mask=torch.ones_like(full_tensor)
                )[0, -1, 0]


            # 自定义奖励计算
            custom_reward = calculate_reward(
                response_text,
                batch_correct_answers[j],
                query_text
            )

            # 加权组合两个奖励
            reward_weight_adapter = 0.5  # reward adapter权重
            reward_weight_custom = 0.5  # 自定义奖励权重

            final_reward = (reward_weight_adapter * reward_score +
                            reward_weight_custom * custom_reward)

            rewards.append(final_reward.detach().clone())

        # PPO更新
        stats = ppo_trainer.step([i for i in query_tensors], [i for i in response_tensors], rewards)
        current_reward = stats['ppo/returns/mean']
        current_score = stats['ppo/mean_scores']
        current_loss = stats['ppo/loss/total']
        # 打印统计信息
        if i % 2 == 0:
            print(f"\nEpoch {epoch + 1}, Batch {i // ppo_config.batch_size + 1}")
            print(f"Mean reward: {np.mean([r.item() for r in rewards]):.4f}")
            if stats:
                print(f"PPO stats: {stats}")

        if current_reward > best_reward and current_score > best_score:
            best_reward = current_reward
            best_score = current_score
            best_output_dir = os.path.join(output_dir, f"epoch_{epoch}_i_{i}")
            ppo_trainer.save_pretrained(best_output_dir)
            print(f"New best model saved at step {i}in epoch {epoch}, reward: {best_reward}, "
                  f"save path: {best_output_dir}. (reward + score)")

# 保存训练后的模型
final_save_path = "/home/bcl/wanghongyu/other/Qwen/RL/output1/ppo_model_6_15_17_12/final"
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)

print(f"PPO trained model saved to {final_save_path}")

# CUDA_VISIBLE_DEVICES=0 nohup python3 -u ppo_for_trl011.py > output1/ppo_for_trl011.log 2>&1 &
# nohup accelerate launch ppo_for_trl011.py > output1/ppo_for_trl011.log 2>&1 &
# 1012211



