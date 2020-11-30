import tensorflow as tf
import numpy as np

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

image_size = (150,150)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "Data DL",
        validation_split=0.28,
        labels='inferred',
        subset="training",
        image_size=image_size,
        seed=1337,
        label_mode='int'
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "Data DL",
        validation_split=0.28,
        labels='inferred',
        subset="validation",
        image_size=image_size,
        seed=1337,
        label_mode='int'
)

# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0

plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

#labels = train_ds.from_tensor_slices(labels)
#print(labels)
#for label in labels:
#    print(label.numpy())

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))#, kernel_regularizer=tf.keras.regularizers.l2(l=0.008)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.04)))
 
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
model.add(layers.Dense(19)) # number of classes (output)

opt= tf.keras.optimizers.Adam(learning_rate=0.0025)
model.compile(optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

history = model.fit(train_ds, epochs=30,
        validation_data=(val_ds))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(val_ds, verbose=2)

print(test_acc)
