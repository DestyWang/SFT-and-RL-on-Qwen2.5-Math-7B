import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
import os
import json
import datetime

# ====== 1. 配置 ======
model_dir = '/home/bcl/wanghongyu/other/Qwen/Qwen2.5-Math-7B-Instruct'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
output_path = '/home/bcl/wanghongyu/other/Qwen/SFT/output'
output_dir = os.path.join(output_path, current_time)
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
system_prompt = ("You are an AI assistant skilled in mathematical reasoning. Please solve the problem using concise step-by-step reasoning process."
                 "Put your final answer within \\boxed{}.")

# ====== 2. 加载模型和分词器 ======
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# 确保tokenizer有pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)

# ====== 3. LoRA配置 ======
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_config)

# ====== 4. 加载数据集 ======
dataset = load_dataset("open-r1/OpenR1-Math-220k", "default")
# if "validation" not in dataset:
#     print("Split validation dataset from training dataset.")
#     split_dataset = dataset["train"].train_test_split(test_size=0.02, seed=42)
#     dataset = DatasetDict({
#         "train": split_dataset["train"],
#         "validation": split_dataset["test"]
#     })


# ====== 5. 数据预处理（SFT格式+COT链条） ======
def preprocess_function(examples):
    processed_inputs = []
    # processed_labels = []

    for problem, solution, answer in zip(examples['problem'], examples['solution'], examples['answer']):
        # 构建完整的对话格式
        answer = str(answer).strip()
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{problem}\n<|assistant|>"
        # full_text = prompt + solution + f"\nIn summary, the final answer is:" + "\\box{" + answer + "}"
        if "\\boxed{" not in solution:
            full_text = prompt + solution + f"\n\nTherefore, the final answer is \\boxed{{{answer}}}."
        else:
            full_text = prompt + solution

        full_tokenized = tokenizer(
            full_text,
            max_length=1024,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )

        # 对prompt部分进行分词以确定需要mask的长度
        prompt_tokenized = tokenizer(
            prompt,
            max_length=1024,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )

        input_ids = full_tokenized["input_ids"]
        attention_mask = full_tokenized["attention_mask"]

        # 创建labels，prompt部分用-100 mask掉
        labels = input_ids.copy()
        prompt_length = len(prompt_tokenized["input_ids"])

        # 确保prompt_length不超过input_ids长度
        if prompt_length < len(labels):
            labels[:prompt_length] = [-100] * prompt_length
        else:
            # 如果prompt长度大于等于总长度，全部mask掉
            labels = [-100] * len(labels)

        processed_inputs.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        })

    # 转换为batch格式
    batch_input_ids = [item["input_ids"] for item in processed_inputs]
    batch_attention_mask = [item["attention_mask"] for item in processed_inputs]
    batch_labels = [item["labels"] for item in processed_inputs]

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing dataset"
)

# ====== 6. 训练参数 ======
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=1e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",  # 使用cosine学习率调度
    warmup_ratio=0.1,  # 添加warmup
    # fp16=True,
    bf16=True,
    logging_steps=10,
    save_total_limit=5,
    report_to=[],
    logging_dir=os.path.join(output_dir, "logs"),
    dataloader_pin_memory=False,  # 避免内存问题,
    label_names=["labels"],
    # 验证参数
    # do_eval=True,
    # load_best_model_at_end=True,
    # metric_for_best_model="eval_loss",
    # greater_is_better=False,
    # eval_strategy="steps",
    # eval_steps=50,
    save_strategy="steps",
    save_steps=100,
)
# ?必须用eval_strategy, 某些版本为evaluation_strategy。。。

# ====== 7. 自定义数据整理器 ======
class CustomDataCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        # 提取各个字段
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # 找到最大长度
        max_len = min(max(len(ids) for ids in input_ids), self.max_length)

        # 手动padding
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, label in zip(input_ids, attention_mask, labels):
            # 截断到max_len
            ids = ids[:max_len]
            mask = mask[:max_len]
            label = label[:max_len]

            # padding
            pad_length = max_len - len(ids)
            if pad_length > 0:
                ids += [self.tokenizer.pad_token_id] * pad_length
                mask += [0] * pad_length
                label += [-100] * pad_length

            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)
            padded_labels.append(label)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long)
        }

# 使用自定义数据整理器
data_collator = CustomDataCollator(tokenizer)

# ====== 8. 自定义早停回调 ======
class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience=5, early_stopping_threshold=0.0):
        super().__init__(early_stopping_patience, early_stopping_threshold)
        self.best_metric = None
        self.wait = 0

    def check_metric_value(self, logs, metric_value):
        # 自定义早停逻辑
        if self.best_metric is None:
            self.best_metric = metric_value
            return False

        if metric_value < self.best_metric - self.early_stopping_threshold:
            self.best_metric = metric_value
            self.wait = 0
        else:
            self.wait += 1

        return self.wait >= self.early_stopping_patience

# ====== 9. Trainer ======
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation", None),
    data_collator=data_collator,
    tokenizer=tokenizer,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    # callbacks=[CustomEarlyStoppingCallback(early_stopping_patience=5)],
)

# ====== 9. 开始训练 ======
if __name__ == "__main__":
    print(f"Training data size: {len(tokenized_datasets['train'])}")
    # print(f"Validation data size: {len(tokenized_datasets.get('validation', None))}")
    print("Starting training...")
    trainer.train()
    print("Training completed. Saving model...")
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    # 保存训练配置
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump({
            "lora_config": {
                "r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "lora_dropout": lora_config.lora_dropout,
                "target_modules": lora_config.target_modules
            },
            "training_args": training_args.to_dict()
        }, f, indent=2)

    print(f"Model and config saved to {output_dir}")

# CUDA_VISIBLE_DEVICES=4 nohup python3 -u sft1.py > output/sft1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1,3,4 nohup accelerate launch --config_file /home/bcl/wanghongyu/.cache/huggingface/accelerate/default_config.yaml sft1.py > output/sft1.log 2>&1 &
# 3309660










