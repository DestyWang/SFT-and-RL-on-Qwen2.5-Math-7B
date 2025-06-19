import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import re
import argparse

# ====== 配置 ======
# model_dir = '/home/bcl/wanghongyu/other/Qwen/SFT/output1/checkpoint-200'
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='/home/bcl/wanghongyu/other/Qwen/SFT/output/checkpoint-900', help='path of your model.')
parser.add_argument('--max_new_tokens', type=int, default=1024)
parser.add_argument('--data_name', type=str, default="open-r1/OpenR1-Math-220k")
args = parser.parse_args()
model_dir = args.model_dir
data_name = args.data_name
# model_dir = '/home/bcl/wanghongyu/other/Qwen/Qwen2.5-Math-7B-Instruct'
system_prompt = ("You are an AI assistant skilled in mathematical reasoning. Please solve the problem using concise step-by-step reasoning process."
                 "Put your final answer within \\boxed{}.")
test_split = "test"
max_new_tokens = args.max_new_tokens
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====== 加载模型和分词器 ======
print(f"Model directory: {model_dir}")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(device)
model.eval()

# ====== 加载并预处理测试集 ======
def lowercase_keys(example):
    """将单个样本的键名转换为小写"""
    return {key.lower(): value for key, value in example.items()}
if data_name == "open-r1/OpenR1-Math-220k":
    dataset = load_dataset("open-r1/OpenR1-Math-220k", "extended")
    if test_split not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.001, seed=42)
        test_dataset = dataset["test"]
    else:
        test_dataset = dataset[test_split]
elif data_name == "agentica-org/DeepScaleR-Preview-Dataset":
    test_dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train[30000:30100]")
elif data_name == "Maxwell-Jia/AIME_2024":
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    test_dataset = dataset.map(lowercase_keys, remove_columns=list(dataset.features.keys()))
else:
    raise ValueError("Invalid data_name")

print(f"Test dataset size: {len(test_dataset)}")

def build_prompt(problem):
    return f"<|system|>\n{system_prompt}\n<|user|>\n{problem}\n<|assistant|>"

def preprocess_batch(batch):
    prompts = [build_prompt(p) for p in batch['problem']]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    return inputs

# ====== 答案抽取增强 ======
def extract_answer(text):
    text = str(text)
    # 优先匹配 \box{} 中的内容
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
    num_match = re.search(r"([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?|[\\d\\w\\(\\)\\+\\-\\*/\\^=]+)", last_line)
    if num_match:
        return num_match.group(0).strip()
    return last_line.strip()

# ====== 评估指标 ======
def pass_at_k(preds, gold, k=1):
    # preds: List[List[str]]，每个样本k个生成
    # gold: List[str]
    correct = 0
    for pred_list, g in zip(preds, gold):
        # 只取前k个生成
        correct += any(extract_answer(p) == extract_answer(g) for p in pred_list[:k])
    return correct / len(gold)

def self_consistency(preds):
    consistencies = []
    for pred_list in preds:
        answers = [extract_answer(p) for p in pred_list]
        if not answers:
            consistencies.append(0)
            continue
        most_common = max(set(answers), key=answers.count)
        consistencies.append(answers.count(most_common) / len(answers))
    return np.mean(consistencies)

def step_accuracy(preds, golds):
    accs = []
    for pred, gold in zip(preds, golds):
        pred_steps = pred.strip().split("\n")
        gold_steps = gold.strip().split("\n")
        match = sum([p.strip() == g.strip() for p, g in zip(pred_steps, gold_steps)])
        accs.append(match / max(len(gold_steps), 1))
    return np.mean(accs)

def path_accuracy(preds, golds):
    accs = []
    for pred, gold in zip(preds, golds):
        pred_path = "\n".join(pred.strip().split("\n")[:-1])
        gold_path = "\n".join(gold.strip().split("\n")[:-1])
        accs.append(pred_path == gold_path)
    return np.mean(accs)

def faithfulness(preds, problems):
    accs = []
    for pred, prob in zip(preds, problems):
        nums = set([s for s in prob.split() if s.replace('.', '', 1).isdigit()])
        accs.append(all(n in pred for n in nums) if nums else True)
    return np.mean(accs)

# ====== 批量生成与评估 ======
def evaluate(model, tokenizer, test_dataset, k=5, batch_size=16):
    all_preds = []
    all_golds = []
    all_golds_sol =[]
    all_problems = []
    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch = test_dataset[i:i+batch_size]
        inputs = preprocess_batch(batch)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        batch_preds = []
        for _ in range(k):
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_preds.append([d.split("<|assistant|>")[-1].strip() for d in decoded])
        batch_preds = list(map(list, zip(*batch_preds)))
        all_preds.extend(batch_preds)
        all_golds.extend([s for s in batch['answer']])
        all_golds_sol.extend([s for s in batch['solution']])
        all_problems.extend([p for p in batch['problem']])
    results = {}
    results['pass@1'] = pass_at_k(all_preds, all_golds, k=1)
    results['pass@5'] = pass_at_k(all_preds, all_golds, k=5)
    results['self_consistency'] = self_consistency(all_preds)
    results['step_accuracy'] = step_accuracy([p[0] for p in all_preds], all_golds_sol)
    results['path_accuracy'] = path_accuracy([p[0] for p in all_preds], all_golds_sol)
    results['faithfulness'] = faithfulness([p[0] for p in all_preds], all_problems)
    print(f"Results: \n{results}")
    return results

if __name__ == "__main__":
    metrics = evaluate(model, tokenizer, test_dataset, k=5, batch_size=16)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

# CUDA_VISIBLE_DEVICES=1 nohup python3 -u eval.py > output2/sft2_eval.log 2>&1 &
# 1212629






