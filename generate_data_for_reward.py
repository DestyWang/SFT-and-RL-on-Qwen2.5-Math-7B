import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
import random
from tqdm import tqdm
import os
from typing import List, Dict, Any, Optional
import logging
import re
import datetime
now = datetime.datetime.now()
print("当前时间:", now.hour, "时", now.minute, "分", now.second, "秒")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardDataGenerator:
    def __init__(
            self,
            model_name: str = "/home/bcl/wanghongyu/other/Qwen/RL/final/reward_model/checkpoint-100",  # 或者你要使用的具体模型
            device: str = "auto",
            max_new_tokens: int = 1024,
            temperature: float = 0.7,
            do_sample: bool = True,
            top_p: float = 0.9,
            system_prompt: str = None,
    ):
        """
        初始化数据生成器

        Args:
            model_name: 基础模型名称
            device: 设备设置
            max_new_tokens: 生成的最大token数
            temperature: 生成温度
            do_sample: 是否采样
            top_p: nucleus sampling参数
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        if system_prompt is None:
            self.system_prompt = ("You are an AI assistant skilled in mathematical reasoning. "
                             "Please solve the problem using concise step-by-step reasoning process. "
                             "Put your final answer within \\boxed{}.")
        else:
            self.system_prompt = system_prompt

        # 加载模型和tokenizer
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        if device != "cuda":
            self.model = self.model.to(device)

        self.device = device
        logger.info(f"Model loaded on device: {device}")

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_answer(self, text: str):
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

    def generate_response(self, problem: str, num_attempts: int = 5) -> List[str]:
        """
        使用基础模型生成回答

        Args:
            problem: 问题文本
            num_attempts: 生成尝试次数

        Returns:
            生成的回答列表
        """
        responses = []

        # 构造promptf"{system_prompt}\n\nProblem: {problem}
        prompt = f"{self.system_prompt}\n\nProblem: {problem}\nSolution:"

        for _ in range(num_attempts):
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # 解码生成的文本
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            responses.append(response)

        return responses

    def create_reward_dataset(
            self,
            dataset_path: str,
            output_path: str = "reward_training_data.json",
            num_samples: Optional[int] = None,
            generation_attempts: int = 1,
            split: str = "train"
    ):
        """
        创建reward模型训练数据

        Args:
            dataset_path: 原始数据集路径或Hugging Face数据集名称
            output_path: 输出文件路径
            num_samples: 处理的样本数量，None表示处理全部
            generation_attempts: 每个问题的生成尝试次数
            split: 数据集分割名称
        """
        # 加载数据集
        logger.info(f"Loading dataset from: {dataset_path}")
        try:
            dataset = load_dataset(dataset_path, split=split)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return

        if num_samples:
            dataset = dataset.select(range(5000, min(5000 + num_samples, len(dataset))))

        logger.info(f"Processing {len(dataset)} samples...")

        reward_data = []
        successful_pairs = 0

        for idx, item in enumerate(tqdm(dataset, desc="Generating reward data")):
            print(f"Current length of reward_data: {len(reward_data)}")
            try:
                # 获取问题和标准答案
                problem = item.get('problem')
                solution = item.get('solution')
                answer = item.get('answer')

                if not problem or not solution:
                    logger.warning(f"Skipping item {idx}: missing problem or solution")
                    continue

                # 提取标准答案
                correct_answer = self.extract_answer(solution)
                if answer != correct_answer:
                    logger.warning(f"Skipping item {idx}: different answer and solution.")
                    continue
                if not correct_answer:
                    logger.warning(f"Skipping item {idx}: cannot extract correct answer")
                    continue

                # 生成候选回答
                generated_responses = self.generate_response(problem, generation_attempts)

                # 筛选出答案错误的回答作为rejected
                rejected_responses = []
                for response in generated_responses:
                    predicted_answer = self.extract_answer(response)
                    if predicted_answer and predicted_answer != correct_answer:
                        rejected_responses.append(response)

                # 如果有错误的回答，创建数据对
                if rejected_responses:
                    # 随机选择一个错误回答作为rejected
                    rejected_response = random.choice(rejected_responses)

                    # 创建数据项
                    data_item = {
                        "chosen": [
                            {"content": self.system_prompt, "role": "system"},
                            {"content": f"Problem: {problem}", "role": "user"},
                            {"content": solution, "role": "assistant"}
                        ],
                        "rejected": [
                            {"content": self.system_prompt, "role": "system"},
                            {"content": f"Problem: {problem}", "role": "user"},
                            {"content": rejected_response, "role": "assistant"}
                        ]
                    }

                    reward_data.append(data_item)
                    successful_pairs += 1

                    if successful_pairs % 100 == 0:
                        logger.info(f"Generated {successful_pairs} reward pairs so far...")
                else:
                    logger.info(f"correct answer for this problem")

            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                continue

        # 保存数据
        if reward_data:
            logger.info(f"Saving {len(reward_data)} reward pairs to {output_path}")
            with open(os.path.join(output_path, "reward_data.json"), 'w', encoding='utf-8') as f:
                json.dump(reward_data, f, ensure_ascii=False, indent=2)

            # 保存数据集统计信息
            stats = {
                "total_pairs": len(reward_data),
                "source_dataset": dataset_path,
                "generation_attempts": generation_attempts,
                "model_used": self.model_name,
                "success_rate": len(reward_data) / len(dataset) if len(dataset) > 0 else 0
            }

            stats_path = os.path.join(output_path, "reward_data_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            logger.info(f"Dataset statistics saved to {stats_path}")
            logger.info(f"Success rate: {stats['success_rate']:.2%}")
        else:
            logger.warning("No valid reward pairs generated!")


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


# 初始化生成器
generator = RewardDataGenerator(
    model_name="/home/bcl/wanghongyu/other/Qwen/SFT/output/checkpoint-900",
    temperature=0.8,
    max_new_tokens=1024
)

# 生成reward训练数据
dataset_path = "agentica-org/DeepScaleR-Preview-Dataset"

generator.create_reward_dataset(
    dataset_path=dataset_path,
    output_path="/home/bcl/wanghongyu/other/Qwen/RL/data/new_reward_data",
    num_samples=10000,
    generation_attempts=1
)

# CUDA_VISIBLE_DEVICES=7 nohup python3 -u generate_data_for_reward.py > output1/data_generation.log 2>&1 &
# 904943


