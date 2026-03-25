import os
import shutil
import random

source_dir = "outputs/rois"
dest_dir = "dataset"

train_ratio = 0.8

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * train_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    for img in train_images:
        train_folder = os.path.join(dest_dir, "train", class_name)
        os.makedirs(train_folder, exist_ok=True)
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_folder, img)
        )

    for img in test_images:
        test_folder = os.path.join(dest_dir, "test", class_name)
        os.makedirs(test_folder, exist_ok=True)
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_folder, img)
        )

print("Dataset split completed successfully.")