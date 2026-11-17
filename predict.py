import tensorflow as tf 
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
import numpy as np 
import os 
import cv2

model = load_model("drone_model.h5") 
print("Model yüklendi!")

class_names = ['airplane', 'bird', 'drone', 'sky']  

def predict_image(img_path):
    original_img = cv2.imread(img_path)

    if original_img is None:
        print("Görüntü okunamadı:", img_path)
        return None, None, None
    
    resized_img = cv2.resize(original_img, (224,224))

    img_array = resized_img / 255.0

    img_array = np.expand_dims(img_array, axis = 0)

    predictions = model.predict(img_array, verbose = 0)
    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]
    return predicted_class, confidence, original_img

test_dir = "dataset/test"  
for real_class in os.listdir(test_dir):
    real_class_path = os.path.join(test_dir, real_class)
    if os.path.isdir(real_class_path): 
        print(f"\nGerçek Class: {real_class}") 
        for img_name in os.listdir(real_class_path):
            img_path = os.path.join(real_class_path, img_name) 
            predicted_class, confidence, display_img = predict_image(img_path) 
            
            if display_img is None:
                continue

            print(f"{img_name} -> Predicted: {predicted_class} (Confidence: {confidence:.2f})")

            text = f"Predicted: {predicted_class} ({confidence*100:.2f}%)"


            if predicted_class == real_class:
                print("Durum: DOGRU")
            else:
                print("Durum: YANLIS")
            
            cv2.imshow("Drone Detection Result", display_img)

            key = cv2.waitKey(0)

            if key == 27:
                cv2.destroyAllWindows()
                exit()

cv2.destroyAllWindows()

print("\nTahmin işlemi tamamlandı.") 
