# Qwen2.5-0.5B Fine-Tuning with LoRA on Databricks Dolly-15k

## Overview
This project fine-tunes **Qwen2.5-0.5B** (Alibaba's 0.5-billion parameter language model) using **LoRA (Low-Rank Adaptation)** on the **Databricks Dolly-15k** instruction-following dataset. Training is designed to run on **Kaggle (GPU T4)**.

## Dataset
**[Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)** — 15,000 high-quality human-generated instruction/response pairs across 8 categories:

| Category | Description |
|---|---|
| `general_qa` | General question answering |
| `open_qa` | Open-ended questions |
| `information_extraction` | Extract info from context |
| `brainstorming` | Creative idea generation |
| `summarization` | Summarize passages |
| `classification` | Classify inputs |
| `closed_qa` | Contextual QA |
| `creative_writing` | Creative text generation |

**Split used:** 8,000 train / 2,000 val / 2,000 test

## Model & Training Setup

| Parameter | Value |
|---|---|
| Base Model | `Qwen/Qwen2.5-0.5B` |
| Method | LoRA (PEFT) |
| LoRA rank `r` | 8 |
| LoRA alpha | 16 |
| Target modules | `q_proj`, `v_proj` |
| Epochs | 10 |
| Batch size | 4 |
| Learning rate | 2e-4 |
| Precision | FP16 (eval in FP32) |
| Max sequence length | 512 |
| Trainable params | ~540K / 494M (0.11%) |

## Evaluation Metrics
- **BLEU Score** — n-gram overlap between prediction and reference
- **ROUGE-1 / ROUGE-2 / ROUGE-L** — recall-oriented overlap metrics
- **Cosine Similarity (per category)** — semantic similarity using `all-MiniLM-L6-v2` sentence embeddings, computed **category-wise** across all 8 Dolly categories
- **Training & Validation Loss** — tracked per epoch

## Results (after 10 epochs)
| Metric | Score |
|---|---|
| Training Loss | ~0.40 |
| Validation Loss | ~0.43 |
| BLEU | ~0.918 |
| ROUGE-L | ~0.948 |

## How to Run (Kaggle)
1. Open `Qwen2.5-0.5B.ipynb` in Kaggle
2. Enable **GPU (T4)** under Settings → Accelerator
3. Enable **Internet** (required to download model + dataset)
4. Run all cells top to bottom (~2 hours on T4)

## Requirements
```
transformers
datasets
peft
accelerate
evaluate
rouge_score
sacrebleu
sentence-transformers
```

## File Structure
```
project 2/
└── Arya/
    ├── Qwen2.5-0.5B.ipynb   ← Main fine-tuning notebook
    └── README.md             ← This file
```

## References
- [Qwen2.5 on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [Databricks Dolly-15k Dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
