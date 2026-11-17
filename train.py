import tensorflow as tf 
from tensorflow.keras import layers, models 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam 
import os 

train_dir = "dataset/train"
img_size = (224, 224) 
batch_size = 16  
epochs = 30      
learning_rate = 1e-4 

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    shear_range=0.1, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest', 
    validation_split=0.2, 
    brightness_range=[0.8,1.2] 
)

train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse', 
    subset='training', 
    shuffle=True 
)

val_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation' 
)

class_names = list(train_dataset.class_indices.keys()) 
num_classes = len(class_names) 
print("Classes:", class_names)

base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')  
base_model.trainable = True  

for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output 
x = layers.GlobalAveragePooling2D()(x) 
x = layers.Dense(128, activation='relu')(x) 
output = layers.Dense(num_classes, activation='softmax')(x) 

model = Model(inputs=base_model.input, outputs=output) 

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'] 
)

model.summary()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# Model Kaydet
model.save("drone_model.h5")
print("Model kaydedildi: drone_model.h5")
