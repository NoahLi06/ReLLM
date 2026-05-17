"""
Stage 3 — PPO Fine-Tuning with RLHF
====================================
Objective:  maximise  E[RM(response)] - β * KL(policy || SFT)

The KL term prevents the policy from drifting so far from the SFT checkpoint
that it learns to "hack" the reward model (e.g. generating very long responses
that superficially look thorough).

Varying β is the key experiment:
  β → 0  : policy chases raw reward, reward hacking likely
  β → ∞  : policy stays at SFT baseline, no alignment benefit
  β ≈ 0.1–0.2 is typical in practice.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, pipeline
from peft import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import os

# ── Config ──────────────────────────────────────────────
SFT_MODEL_DIR    = "./sft-model"
REWARD_MODEL_DIR = "./reward-model"
OUTPUT_DIR       = "./ppo-model"
DEVICE           = "mps"
NUM_EXAMPLES     = 200          # PPO is slow; keep small for MPS
MAX_INPUT_LEN    = 128          # prompt length fed to the policy
MAX_NEW_TOKENS   = 128          # max tokens the policy generates per prompt
BATCH_SIZE       = 4
MINI_BATCH_SIZE  = 1
PPO_EPOCHS       = 4            # gradient steps per PPO update
KL_COEF          = 0.2          # β — the alignment-capability tradeoff knob
LEARNING_RATE    = 1e-5
# ────────────────────────────────────────────────────────

# ── Load tokenizer ───────────────────────────────────────
base = SFT_MODEL_DIR if os.path.isdir(SFT_MODEL_DIR) else "HuggingFaceTB/SmolLM2-1.7B"
print(f"Loading policy from: {base}")

tokenizer = AutoTokenizer.from_pretrained(base)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"     # left-pad for generation

# ── Policy model (SFT checkpoint + value head) ──────────
# AutoModelForCausalLMWithValueHead wraps a causal LM and adds a scalar
# value head used by PPO to estimate V(s) (the baseline).
policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    base,
    torch_dtype=torch.float32,
    device_map={"": DEVICE},
)

# ── Reference model (frozen SFT checkpoint) ─────────────
# Used only to compute the KL divergence term.  Must stay frozen.
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    base,
    torch_dtype=torch.float32,
    device_map={"": DEVICE},
)
for p in ref_model.parameters():
    p.requires_grad_(False)

# ── Reward model ─────────────────────────────────────────
rm_base = REWARD_MODEL_DIR if os.path.isdir(REWARD_MODEL_DIR) else "HuggingFaceTB/SmolLM2-1.7B"
print(f"Loading reward model from: {rm_base}")

reward_tokenizer = AutoTokenizer.from_pretrained(rm_base)
reward_tokenizer.pad_token = reward_tokenizer.eos_token
reward_tokenizer.padding_side = "right"

reward_model = AutoModelForSequenceClassification.from_pretrained(
    rm_base,
    num_labels=1,
    torch_dtype=torch.float32,
    device_map={"": DEVICE},
    ignore_mismatched_sizes=True,
)
reward_model.eval()

def get_reward(texts: list[str]) -> list[torch.Tensor]:
    """Return a list of scalar reward tensors, one per text."""
    enc = reward_tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN + MAX_NEW_TOKENS,
        padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        scores = reward_model(**enc).logits.squeeze(-1)   # (batch,)
    return [scores[i] for i in range(scores.size(0))]

# ── Dataset — extract prompts only ──────────────────────
# For PPO we only need the prompt; the policy generates the response.
# We strip to the human turn so the policy sees a natural prompt.
print("Loading dataset...")
raw = load_dataset("Anthropic/hh-rlhf", split=f"train[:{NUM_EXAMPLES}]")

def extract_prompt(example):
    """Keep only the last Human turn as the prompt."""
    text = example["chosen"]
    # hh-rlhf format: "\n\nHuman: ...\n\nAssistant: ..."
    # Take everything up to (and including) the last "Assistant:" prefix.
    if "\n\nAssistant:" in text:
        prompt = text.rsplit("\n\nAssistant:", 1)[0] + "\n\nAssistant:"
    else:
        prompt = text[:256]
    return {"query": prompt}

dataset = raw.map(extract_prompt, remove_columns=["chosen", "rejected"])

def tokenize_query(example):
    ids = tokenizer(
        example["query"],
        truncation=True,
        max_length=MAX_INPUT_LEN,
        padding=False,
    )
    example["input_ids"] = ids["input_ids"]
    return example

dataset = dataset.map(tokenize_query)
dataset.set_format(type="torch")
print(f"Dataset ready: {len(dataset)} prompts")

# ── PPO config ───────────────────────────────────────────
ppo_config = PPOConfig(
    model_name=base,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    mini_batch_size=MINI_BATCH_SIZE,
    ppo_epochs=PPO_EPOCHS,
    kl_penalty="kl",       # subtract β·KL from the reward
    init_kl_coef=KL_COEF,  # β
    adap_kl_ctrl=False,    # fixed β — set True to auto-tune
    gradient_accumulation_steps=1,
    optimize_device_cache=True,
    log_with=None,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=lambda data: {
        "input_ids": [d["input_ids"] for d in data],
        "query":     [d["query"]     for d in data],
    },
)

# ── Generation kwargs ────────────────────────────────────
gen_kwargs = {
    "min_new_tokens": 16,
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

# ── PPO training loop ────────────────────────────────────
print("Starting PPO training...")
print(f"  KL coefficient β = {KL_COEF}  (try 0.0, 0.05, 0.2, 0.5 to observe tradeoff)")

for step, batch in enumerate(ppo_trainer.dataloader):
    query_tensors = batch["input_ids"]   # list of 1-D tensors

    # 1. Policy generates responses
    response_tensors = ppo_trainer.generate(
        query_tensors,
        return_prompt=False,
        **gen_kwargs,
    )

    # 2. Decode full text (prompt + response) for the reward model
    full_texts = [
        tokenizer.decode(q.tolist() + r.tolist(), skip_special_tokens=True)
        for q, r in zip(query_tensors, response_tensors)
    ]

    # 3. Score with reward model
    rewards = get_reward(full_texts)

    # 4. PPO update
    #    Internally: advantage = reward - β*KL - V(s)
    #                policy gradient + value function regression
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if step % 10 == 0:
        mean_reward = torch.stack(rewards).mean().item()
        mean_kl     = stats.get("objective/kl", float("nan"))
        print(f"Step {step:4d} | mean reward: {mean_reward:.4f} | mean KL: {mean_kl:.4f}")

# ── Save ─────────────────────────────────────────────────
print("Saving PPO-tuned model...")
ppo_trainer.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"PPO model saved to {OUTPUT_DIR}")
print()
print("Experiment suggestion:")
print("  Re-run with KL_COEF in [0.0, 0.05, 0.1, 0.2, 0.5] and plot mean_reward vs KL.")
print("  At low β you should observe reward rising but KL exploding (reward hacking).")
print("  At high β reward stays near the SFT baseline — the model barely changes.")
