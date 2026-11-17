import os
import requests

classes = ["bird", "airplane", "sky"]
train_dir = "dataset/train"
num_images = 50  # Her sınıf için

urls = {
    "bird": ["https://images.unsplash.com/photo-1570111863113-0c6d5ed9d02c"]*num_images,
    "airplane": ["https://images.unsplash.com/photo-1526512340740-9b7fc1d3e982"]*num_images,
    "sky": ["https://images.unsplash.com/photo-1506744038136-46273834b3fb"]*num_images
}


for cls in classes:
    cls_path = os.path.join(train_dir, cls)
    os.makedirs(cls_path, exist_ok=True)
    
    for i, url in enumerate(urls[cls]):
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(cls_path, f"{cls}_{i}.jpg"), "wb") as f:
                f.write(response.content)
        print(f"{cls}_{i}.jpg indirildi.")
