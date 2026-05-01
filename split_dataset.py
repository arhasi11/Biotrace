import os
import shutil
import random

source = "dataset"   # images are directly in A, B, AB, O
train_dir = "dataset/train"
test_dir = "dataset/test"

labels = ["A","B","AB","O"]

for label in labels:
    path = os.path.join(source, label)

    if not os.path.exists(path):
        print(f"❌ Missing folder: {path}")
        continue

    files = os.listdir(path)
    random.shuffle(files)

    split = int(0.8 * len(files))

    train_files = files[:split]
    test_files = files[split:]

    for f in train_files:
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        shutil.copy(os.path.join(path, f), os.path.join(train_dir, label, f))

    for f in test_files:
        os.makedirs(os.path.join(test_dir, label), exist_ok=True)
        shutil.copy(os.path.join(path, f), os.path.join(test_dir, label, f))

print("✅ Dataset split correctly")