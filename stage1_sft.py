import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"
# MPS support can be compiled into PyTorch but unavailable at runtime (for
# example on a non-Apple worker).  Select a usable accelerator instead of
# failing during model loading.
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
OUTPUT_DIR = "./sft-model"
NUM_EXAMPLES = 1000
MAX_LENGTH = 256
# ────────────────────────────────────────────────────────

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=MODEL_DTYPE,
)
model.to(DEVICE)
print(f"Model loaded on {DEVICE}")

# LoRA: train only a small fraction of parameters
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

print("Loading dataset...")
dataset = load_dataset("Anthropic/hh-rlhf", split=f"train[:{NUM_EXAMPLES}]")

# SFT trains on chosen responses only — these are the "good" examples
def extract_chosen(example):
    return {"text": example["chosen"]}

dataset = dataset.map(extract_chosen, remove_columns=["chosen", "rejected"])
print(f"Dataset ready: {len(dataset)} examples")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,   # effective batch size = 4
    learning_rate=2e-4,
    fp16=False,                      # MPS doesn't support fp16 training
    bf16=False,
    logging_steps=10,
    save_steps=200,
    dataloader_num_workers=0,        # MPS requires workers=0
    max_length=256,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

print("Starting SFT training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
