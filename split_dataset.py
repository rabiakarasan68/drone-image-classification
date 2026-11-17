import os
import random
import shutil

split_ratio = 0.8  

base_dir = "dataset"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

for class_name in os.listdir(train_dir):

    class_train_path = os.path.join(train_dir, class_name)

    if not os.path.isdir(class_train_path):
        continue

    class_test_path = os.path.join(test_dir, class_name)

    if not os.path.exists(class_test_path):
        os.makedirs(class_test_path)

    files = [f for f in os.listdir(class_train_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    random.shuffle(files)

    split_index = int(len(files) * split_ratio)

    test_files = files[split_index:]

    for file in test_files:

        src = os.path.join(class_train_path, file)
        dst = os.path.join(class_test_path, file)

        shutil.move(src, dst)

    print(f"{class_name}: {len(test_files)} files moved to test")

print("Dataset split completed!")
