import random
import os

DATA_DIR = os.path.expanduser(~/data/aimo_cleaned_data_v1)
SEPARATOR = = * 50
VAL_RATIO = 0.1
SEED = 42

files = [
    aops_cleaned.txt,
    hmmt_cleaned.txt,
    math_stackexchange_cleaned.txt,
    similar_sources_cleaned.txt,
]

all_entries = []
for fname in files:
    path = os.path.join(DATA_DIR, fname)
    with open(path) as f:
        content = f.read()
    entries = [e.strip() for e in content.split(SEPARATOR) if e.strip()]
    print(f"{fname}: {len(entries)} entries")
    all_entries.extend(entries)

print(f"Total: {len(all_entries)} entries")

random.seed(SEED)
random.shuffle(all_entries)

n_val = max(1, round(len(all_entries) * VAL_RATIO))
val_entries = all_entries[:n_val]
train_entries = all_entries[n_val:]

print(f"Train: {len(train_entries)} | Val: {len(val_entries)}")

sep = f"\n\n{SEPARATOR}\n\n"

train_path = os.path.join(DATA_DIR, "train.txt")
with open(train_path, "w") as f:
    f.write(sep.join(train_entries))

val_path = os.path.join(DATA_DIR, "val.txt")
with open(val_path, "w") as f:
    f.write(sep.join(val_entries))

print(f"Saved: {train_path}")
print(f"Saved: {val_path}")
