# 🔢 Qwen2.5-Math-7B: Supervised Fine-tuning and PPO-based Reinforcement Learning

### This is our final project for Natural Language Understanding course. 

This project explores **fine-tuning and reinforcement learning** techniques to enhance the mathematical reasoning capabilities of the large language model [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct). We use **supervised fine-tuning (SFT)** on OpenR1-Math-220k and **reinforcement learning (RLHF with PPO)** on DeepScaleR to improve performance on downstream mathematical tasks.

## 📌 Project Highlights

- ✅ **Supervised fine-tuning** with [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- ✅ **Reward modeling** and **PPO training** using [DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)
- ✅ Performance evaluated using `pass@1`, `pass@5`, `self-consistency`
- ✅ Achieved +11.3% improvement on `pass@1` (25.0% → 36.3%) on OpenR1 dataset
- 🧠 Adapter-based PPO implementation using **LoRA** for all model heads to reduce memory footprint

---

## ⚙️ Dependencies

This project relies on the HuggingFace ecosystem. There are two sets of dependencies depending on the training type:

### 🔹 For Supervised Fine-tuning (SFT) and evaluation
- `transformers==4.52.4`
- `trl==0.18.1`
- `datasets==3.6.0`
- `peft==0.15.2`
- `accelerate==1.7.0`
- `torch==2.7.1`

### 🔸 For PPO Reinforcement Learning
- `transformers==4.52.4`
- `trl==0.11.0`
- `datasets==3.6.0`
- `peft==0.15.2`
- `accelerate==1.7.0`
- `torch==2.7.1`

## 🧪 Evaluation
Use the following command to evaluate a fine-tuned checkpoint:
```bash
python3 -u eval.py \
  --model_dir MODLE_dir \
  --max_new_tokens 1024 \
  --data_name DATA_NAME
```


---

## 📚 Datasets

### 1. [OpenR1-Math-220k (default split)](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- 94k math problems with 2–4 CoT traces per problem
- Problems sourced from **NuminaMath 1.5** and verified using Math Verify and LLaMA-3.3-70B

### 2. [DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)
- ~40k problems from AIME, AMC, OmniMATH, etc.
- Used for **reward model training** and **PPO**

---

## 🧪 Methodology

### 🔧 Supervised Fine-tuning (SFT)
- LoRA-based fine-tuning of Qwen2.5-Math-7B
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Loss: CrossEntropy over CoT + final answer

| Hyperparameter     | Value       |
|--------------------|-------------|
| LoRA rank          | 16          |
| α (LoRA scale)     | 32          |
| Epochs             | 2           |
| Learning rate      | 1e-5        |
| Max length         | 1024 tokens |
| Scheduler          | Cosine      |
| Precision          | bf16        |

### 🧠 Reinforcement Learning with PPO
- Uses **SFT checkpoint** as base policy
- Reward model trained on ~1000 `chosen` vs `rejected` pairs (custom generated)
- Final reward = weighted sum of reward model + handcrafted metrics

**Custom reward includes:**
- +1 for correct answer
- length penalty for long outputs
- +0.2 if box notation `\boxed{}` used
- +0.2 if keywords like "step", "then", "therefore" appear

| PPO Config          | Value         |
|---------------------|---------------|
| LoRA rank           | 8             |
| α                   | 16            |
| Epochs              | 3             |
| Top-p sampling      | 0.9           |
| Max length          | 1024          |
| Adapter strategy    | 4-way switch (policy, value, ref, reward) via LoRA |

---

## 📈 Results

### 📌 pass@1

| Dataset             | Base | SFT   | RL    |
|---------------------|------|-------|-------|
| OpenR1-Math         | 0.2500 | 0.3030 | **0.3636** |
| DeepScaleR          | 0.2200 | 0.1400 | **0.2700** |

### 📌 pass@5

| Dataset             | Base | SFT   | RL    |
|---------------------|------|-------|-------|
| OpenR1-Math         | 0.4848 | 0.5076 | **0.5152** |
| DeepScaleR          | 0.4600 | 0.4600 | **0.4700** |

### 📌 self-consistency

| Dataset             | Base | SFT   | RL    |
|---------------------|------|-------|-------|
| OpenR1-Math         | 0.5773 | 0.5152 | **0.6424** |
| DeepScaleR          | 0.6240 | 0.3680 | **0.5240** |

---

## 💡 Observations

- SFT improves `pass@1` significantly but leads to slight distribution skew (decreased diversity and self-consistency)
- PPO recovers performance and diversity on OpenR1 while partially mitigating SFT-induced knowledge forgetting
- Model still struggles on very challenging datasets (e.g., AIME2024), indicating limits of model scale

---

## 💬 Citation

If you find this project helpful, feel free to cite:

```bibtex
@misc{qwen2.5-math-sft-ppo,
  title={Fine-tuning and RLHF for Qwen2.5-Math-7B},
  author={Zhaorun Chen and Hongyu Wang},
  institution={Qiuzhen College, Tsinghua University},
  year={2025},
  note={Final project for NLP Course}
}

