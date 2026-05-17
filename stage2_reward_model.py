import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List

# ── Config ──────────────────────────────────────────────
SFT_MODEL_DIR = "./sft-model"       # trained in stage 1
MODEL_NAME    = "HuggingFaceTB/SmolLM2-1.7B"  # fallback if SFT model missing
DEVICE        = "mps"
OUTPUT_DIR    = "./reward-model"
NUM_EXAMPLES  = 1000
MAX_LENGTH    = 256
# ────────────────────────────────────────────────────────

# ── Load tokenizer and base model ───────────────────────
# We initialise a sequence-classification head (single scalar output = reward score).
# num_labels=1 gives one logit per sequence — that logit IS the reward.
import os
base = SFT_MODEL_DIR if os.path.isdir(SFT_MODEL_DIR) else MODEL_NAME
print(f"Loading reward model base from: {base}")

tokenizer = AutoTokenizer.from_pretrained(base)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForSequenceClassification.from_pretrained(
    base,
    num_labels=1,
    torch_dtype=torch.float32,   # float32 for stable classification head training
    device_map={"": DEVICE},
    ignore_mismatched_sizes=True,
)
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA on the backbone — keeps the classification head fully trainable
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.SEQ_CLS,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Dataset ──────────────────────────────────────────────
print("Loading dataset...")
raw = load_dataset("Anthropic/hh-rlhf", split=f"train[:{NUM_EXAMPLES}]")

def tokenize_pair(example):
    """Tokenise chosen and rejected into separate fields."""
    chosen  = tokenizer(example["chosen"],  truncation=True, max_length=MAX_LENGTH, padding="max_length")
    rejected = tokenizer(example["rejected"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    return {
        "input_ids_chosen":      chosen["input_ids"],
        "attention_mask_chosen": chosen["attention_mask"],
        "input_ids_rejected":      rejected["input_ids"],
        "attention_mask_rejected": rejected["attention_mask"],
    }

dataset = raw.map(tokenize_pair, remove_columns=["chosen", "rejected"])
dataset = dataset.train_test_split(test_size=0.05)
print(f"Train: {len(dataset['train'])}  Eval: {len(dataset['test'])}")

# ── Bradley-Terry loss ───────────────────────────────────
# Loss = -log σ(r_chosen - r_rejected)
# The model should assign a higher scalar reward to the preferred response.
@dataclass
class RewardDataCollator:
    """Collates paired (chosen, rejected) examples into a single batch dict."""
    tokenizer: Any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            "input_ids":      torch.tensor([f["input_ids_chosen"]      for f in features]),
            "attention_mask": torch.tensor([f["attention_mask_chosen"]  for f in features]),
            "input_ids_rejected":      torch.tensor([f["input_ids_rejected"]      for f in features]),
            "attention_mask_rejected": torch.tensor([f["attention_mask_rejected"] for f in features]),
        }
        return batch


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Score chosen responses
        rewards_chosen = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).logits.squeeze(-1)          # (batch,)

        # Score rejected responses
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
        ).logits.squeeze(-1)          # (batch,)

        # Bradley-Terry: -log σ(r_w - r_l)
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
        return loss


# ── Training ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    fp16=False,
    bf16=False,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    dataloader_num_workers=0,
    remove_unused_columns=False,     # our collator uses custom keys
    label_names=[],                  # no conventional labels; loss is computed manually
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=RewardDataCollator(tokenizer),
)

print("Starting Reward Model training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Reward model saved to {OUTPUT_DIR}")
