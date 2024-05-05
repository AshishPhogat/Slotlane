import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

dataset_path = "../clf-data"
train_ds = data_gen.flow_from_directory(
    directory=dataset_path,
    subset="training",
    seed=123,
    target_size=(29,68),
    batch_size=32,
    class_mode='sparse',
    shuffle=True)

val_ds = data_gen.flow_from_directory(
    directory=dataset_path,
    subset="validation",
    seed=123,
    target_size=(29, 68),
    batch_size=32,
    class_mode='sparse',
    shuffle=True)
    
    
    
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(29, 68, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')])
    
    
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
            
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

model.save('./trained_model.keras')