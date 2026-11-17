
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
import seaborn as sns 
import matplotlib.pyplot as plt 
import os


test_dir = "dataset/test" 
img_size = (224, 224) 
batch_size = 32 


model = load_model("drone_model.h5") 
print("Model yüklendi.")


test_datagen = ImageDataGenerator(rescale=1./255) 

test_dataset = test_datagen.flow_from_directory( 
    test_dir, 
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse', 
    shuffle=False   #
)

class_names = list(test_dataset.class_indices.keys()) 

print("Classes:", class_names)


predictions = model.predict(test_dataset)

y_pred = np.argmax(predictions, axis=1) 
y_true = test_dataset.classes 


accuracy = accuracy_score(y_true, y_pred) 

print("\nAccuracy:", accuracy)


print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)


plt.figure(figsize=(8,6)) 

sns.heatmap( 
    cm,
    annot=True, 
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.show()
