import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import EfficientNetB0
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model, save_model

def efficientnet_model(input_shape, num_classes):
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet', drop_connect_rate=0.4)

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

input_shape_efficientnet = (32, 32, 3)
num_classes_efficientnet = 10

model_efficientnet = efficientnet_model(input_shape_efficientnet, num_classes_efficientnet)

model_efficientnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_efficientnet.summary()

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory("C:/Users/91900/Desktop/CORE PROJECT/Dataset/Train",
                                                    target_size=input_shape_efficientnet[:2],
                                                    batch_size=32, class_mode='sparse')
print(f"Number of training batches: {len(train_generator)}")
print(f"Number of training samples: {len(train_generator.classes)}")

test_generator = test_datagen.flow_from_directory("C:/Users/91900/Desktop/CORE PROJECT/Dataset/Test",
                                                  target_size=input_shape_efficientnet[:2],
                                                  batch_size=32, class_mode='sparse')
print(f"Number of testing batches: {len(test_generator)}")
print(f"Number of testing samples: {len(test_generator.classes)}")

if len(train_generator) == 0 or len(test_generator) == 0:
    print("No data found. Check your directory paths and data.")
else:

    history = model_efficientnet.fit(train_generator, epochs=10, validation_data=test_generator)

    model_efficientnet.save("C:/Users/91900/Desktop/CORE PROJECT/EfficientNet.h5")

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy Across Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    accuracy_efficientnet = model_efficientnet.evaluate(test_generator)[1]
    print(f"EfficientNet Accuracy on Testing Set: {accuracy_efficientnet * 100:.2f}%")
