"""Stage 3 — PPO fine-tuning with RLHF using TRL's current experimental PPO API.

Objective: maximise E[reward(response)] - beta * KL(policy || reference).
Run stage 1 and stage 2 first. Stage 1 is optional (the base model is used if
no valid SFT adapter is present), but a trained reward-model checkpoint is
required: PPO against an untrained reward head has no meaningful objective.
"""

import os

# PPO is experimental in TRL 1.4. Silence its import-time warning because this
# script intentionally uses that API until PPO is promoted to TRL's stable API.
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, AutoPeftModelForSequenceClassification, PeftConfig
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl.experimental.ppo import PPOConfig, PPOTrainer


# ── Config ──────────────────────────────────────────────────────────────────
BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B"
SFT_MODEL_DIR = "./sft-model"
REWARD_MODEL_DIR = "./reward-model"
OUTPUT_DIR = "./ppo-model"
NUM_EXAMPLES = 200
MAX_INPUT_LEN = 128
MAX_NEW_TOKENS = 128
BATCH_SIZE = 4
MINI_BATCH_SIZE = 1
PPO_EPOCHS = 4
KL_COEF = 0.2
LEARNING_RATE = 1e-5
SEED = 42


def has_checkpoint(path: str) -> bool:
    """Return whether *path* contains a Hugging Face model or PEFT adapter."""
    if not os.path.isdir(path):
        return False

    files = set(os.listdir(path))
    has_adapter = "adapter_config.json" in files and bool(
        {"adapter_model.safetensors", "adapter_model.bin"} & files
    )
    has_model = "config.json" in files and bool(
        {
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        }
        & files
    )
    return has_adapter or has_model


def checkpoint_or_base(path: str, fallback: str) -> str:
    """Use a valid local checkpoint; treat missing or empty directories as absent."""
    return path if has_checkpoint(path) else fallback


def is_adapter_checkpoint(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def adapter_base(path: str) -> str:
    """Resolve a PEFT adapter to the transformer checkpoint beneath it."""
    return PeftConfig.from_pretrained(path).base_model_name_or_path


def load_policy(path: str, *, trainable: bool):
    if is_adapter_checkpoint(path):
        return AutoPeftModelForCausalLM.from_pretrained(
            path, is_trainable=trainable, dtype=torch.float32
        )
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)


def load_reward_model(path: str):
    if is_adapter_checkpoint(path):
        return AutoPeftModelForSequenceClassification.from_pretrained(path, dtype=torch.float32)
    return AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=1, dtype=torch.float32, ignore_mismatched_sizes=True
    )


# ── Models ──────────────────────────────────────────────────────────────────
# The current PPOTrainer owns device placement through Accelerate. Do not pass
# device_map here: Accelerate cannot train models that were pre-sharded with it.
policy_source = checkpoint_or_base(SFT_MODEL_DIR, BASE_MODEL)
if policy_source == BASE_MODEL:
    print("No valid SFT checkpoint found; starting PPO from the base model.")
else:
    print(f"Loading SFT policy from: {policy_source}")

if not has_checkpoint(REWARD_MODEL_DIR):
    raise FileNotFoundError(
        f"No trained reward-model checkpoint found in {REWARD_MODEL_DIR!r}. "
        "Run stage2_reward_model.py before PPO training."
    )

tokenizer = AutoTokenizer.from_pretrained(policy_source)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

policy = load_policy(policy_source, trainable=True)
ref_model = load_policy(policy_source, trainable=False)
for parameter in ref_model.parameters():
    parameter.requires_grad_(False)

# TRL 1.4 PPO uses separate sequence-classification models for the reward and
# value heads. Both expose the `score` module expected by PPOTrainer.
reward_model = load_reward_model(REWARD_MODEL_DIR)
reward_model.config.pad_token_id = tokenizer.pad_token_id

# A causal-LM adapter has no sequence-classification head, so initialise the
# PPO critic from its underlying transformer checkpoint instead of the adapter.
value_source = adapter_base(policy_source) if is_adapter_checkpoint(policy_source) else policy_source
value_model = AutoModelForSequenceClassification.from_pretrained(
    value_source,
    num_labels=1,
    dtype=torch.float32,
    ignore_mismatched_sizes=True,
)
value_model.config.pad_token_id = tokenizer.pad_token_id

for name, model in (("reward model", reward_model), ("value model", value_model)):
    if not hasattr(model, "score"):
        raise TypeError(f"The {name} must expose a sequence-classification `score` head.")


# ── Dataset ─────────────────────────────────────────────────────────────────
def extract_prompt(example: dict) -> dict:
    """Keep the dialogue through the final Assistant prefix as a policy prompt."""
    chosen = example["chosen"]
    if "\n\nAssistant:" in chosen:
        prompt = chosen.rsplit("\n\nAssistant:", 1)[0] + "\n\nAssistant:"
    else:
        prompt = chosen[:MAX_INPUT_LEN]
    return {"query": prompt}


def tokenize_prompt(example: dict) -> dict:
    return tokenizer(example["query"], truncation=True, max_length=MAX_INPUT_LEN)


print("Loading PPO prompts...")
dataset = load_dataset("Anthropic/hh-rlhf", split=f"train[:{NUM_EXAMPLES}]")
dataset = dataset.map(extract_prompt, remove_columns=["chosen", "rejected"])
dataset = dataset.map(tokenize_prompt, remove_columns=["query"])
dataset = dataset.train_test_split(test_size=0.05, seed=SEED)
print(f"Train: {len(dataset['train'])}  Eval: {len(dataset['test'])}")


# ── PPO ─────────────────────────────────────────────────────────────────────
# New TRL config derives the global batch size from per-device batch size and
# gradient accumulation. These values preserve the old script's 4 examples per
# PPO update and one example per minibatch.
ppo_config = PPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=MINI_BATCH_SIZE,
    per_device_eval_batch_size=MINI_BATCH_SIZE,
    gradient_accumulation_steps=BATCH_SIZE,
    num_mini_batches=BATCH_SIZE // MINI_BATCH_SIZE,
    num_train_epochs=1,
    num_ppo_epochs=PPO_EPOCHS,
    response_length=MAX_NEW_TOKENS,
    temperature=0.7,
    stop_token="eos",
    kl_coef=KL_COEF,
    total_episodes=len(dataset["train"]),
    logging_steps=1,
    eval_strategy="no",
    save_strategy="epoch",
    report_to="none",
    bf16=False,
    fp16=False,
    gradient_checkpointing=False,
    seed=SEED,
)

trainer = PPOTrainer(
    args=ppo_config,
    processing_class=tokenizer,
    model=policy,
    ref_model=ref_model,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

print("Starting PPO training...")
print(f"  KL coefficient beta = {KL_COEF}")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"PPO policy saved to {OUTPUT_DIR}")
print("Try KL_COEF values 0.0, 0.05, 0.1, 0.2, and 0.5, then compare reward and KL logs.")
