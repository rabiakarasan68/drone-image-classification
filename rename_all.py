import os

base_dir = "dataset/train"

for class_name in os.listdir(base_dir):

    class_path = os.path.join(base_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"Renaming files in: {class_name}")

    count = 0

    for filename in os.listdir(class_path):

        old_path = os.path.join(class_path, filename)

        # uzantıyı al
        ext = os.path.splitext(filename)[1].lower()

        # sadece resim dosyaları
        if ext not in [".jpg", ".jpeg", ".png"]:
            continue

        new_name = f"{class_name}_{count}{ext}"

        new_path = os.path.join(class_path, new_name)

        os.rename(old_path, new_path)

        count += 1

print("All files renamed successfully!")
