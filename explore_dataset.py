from datasets import load_dataset

ds = load_dataset("Anthropic/hh-rlhf", split="train")

print(f"Dataset size: {len(ds)}")
print(f"Columns: {ds.column_names}")
print("\n" + "="*60)
print("EXAMPLE 1 — CHOSEN (preferred response):")
print("="*60)
print(ds[0]["chosen"])
print("\n" + "="*60)
print("EXAMPLE 1 — REJECTED (less preferred response):")
print("="*60)
print(ds[0]["rejected"])