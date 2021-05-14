import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (227,227))
    return image, label

# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []

image_size = (150,150)#150
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "NuevaData",
        #validation_split=0.28,
        labels='inferred',
        #subset="training",
        image_size=image_size,
        batch_size=129,
        seed=2327,
        label_mode='int'
)

lbls = np.array([])
for element, labels in train_ds.as_numpy_iterator():
    if(len(lbls) == 0):
        lbls = labels[:]
    else:
        lbls = np.append(lbls, labels)

print("labels: ", lbls)
print("label size:", len(lbls))
#val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#        "NuevaData",
#        validation_split=0.28,
#        labels='inferred',
#        subset="validation",
#        image_size=image_size,
#        seed=1337,
#        label_mode='int'
#)

skf = StratifiedKFold(n_splits = 5, random_state = 7, shuffle = True)

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
#validation_ds_size = tf.data.experimental.cardinality(val_ds).numpy()
print("Training data size: ", train_ds_size)

train_ds_an = (train_ds.map(process_images).shuffle(buffer_size=train_ds_size).batch(batch_size=32, drop_remainder=True))

#validation_ds_an =(val_ds
#        .map(process_images)
#        .shuffle(buffer_size=validation_ds_size)
#        .batch(batch_size=32, drop_remainder=True))

model_an = models.Sequential([
    layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)), #0
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"), #3
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    layers.Conv2D(filters=384, kernel_size=(3,3), strides=(2,2), activation='relu', padding="same"), #6
    layers.BatchNormalization(),
    layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"), #8
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"), #10
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(19, activation='softmax')
    ])

# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0

plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        #plt.imshow(tf.image.per_image_standardization(images[i]).numpy().astype("float32"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

#for images, labels in val_ds.take(1):
#    for i in range(9):
#        ax = plt.subplot(3, 3, i+1)
#        plt.imshow(images[i].numpy().astype("uint8"))
#        plt.title(int(labels[i]))
#        plt.axis("off")
#plt.show()

#labels = train_ds.from_tensor_slices(labels)
#print(labels)
#for label in labels:
#    print(label.numpy())
result, history, ixs_trn, ixs_tst = [], [], [], []
for images, labels in train_ds.take(-1):
    np_images = images.numpy()
    np_labels = labels.numpy()
    print(len(np_images))
    print(len(np_labels))
    for train_index, test_index in skf.split(np_images, np_labels):
        ixs_trn.append(train_index)
        ixs_tst.append(test_index)
        #train = train_ds.enumerate().filter(lambda x: x in train_inde, self)
        #val = train_ds.enumerate().filter(lambda x: x in test_index, self)
        train_images = np_images[train_index]
        train_labels = np_labels[train_index]
        val_images = np_images[test_index]
        val_labels = np_labels[test_index]

        file_path = "weights_base.ckpt"
        if not os.path.exists(file_path):
            with open(file_path, 'w') as fp:
                pass
            fp.close() 
        checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
        early = EarlyStopping(monitor='val_loss', mode='min', patience=8)

        callbacks_list = [checkpoint, early]

        model = models.Sequential()

        model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.008)))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))#, kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
        model.add(layers.MaxPooling2D((2,2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
        model.add(layers.MaxPooling2D((2,2)))

        #model.add(layers.Conv2D(64, (3,3), activation='relu'))
        #model.add(layers.MaxPooling2D((2,2)))
        #model.summary()

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))#, kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(1, activation='sigmoid')) # number of classes (output)
        model.summary()

        opt= tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=opt,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

        hist = model.fit(train_images, train_labels, epochs=30, callbacks=callbacks_list, validation_data=(val_images, val_labels))
        history.append(hist)

        model.load_weights(file_path)
        #img = load_img('./NuevaData/No_sanos/Captura CARIES  SEVERISIMMA.PNG', target_size=(150,150))
        #convert the image to an array
        #img = img_to_array(img)
        #expand dimensions so that it represents a single 'sample'
        #img = np.expand_dims(img, axis=0)

        result.append(model.evaluate(val_images, val_labels))

print(model.metrics_names)
print("size of history: ", len(history))
print("size of train indexes: ", len(ixs_trn))
print("size of test indexes: ", len(ixs_tst))
print(result)
#predictions =np.array([])
#labels = np.array([])
#for x, y in train_ds:
    #predictions = np.concatenate([predictions, np.argmax(model.predict_classes(x), axis=-1)])
    #labels = np.append(labels, np.argmax(y.numpy(), axis=-1))
    #print("asd: ", [np.argmax(y.numpy(), axis=-1)])

#print(labels)
#quit()
#model_an.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
#model_an.summary()

#history_an = model_an.fit(train_ds,
#        epochs=30,
#        validation_data=val_ds,
#        validation_freq=1)

#history = model.fit(train_ds, epochs=30,
#        validation_data=(val_ds))

def history_plot(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

def history_multiplot(history, train_size, n_split):
    history_ixs = 0
    if(train_size==1):
        fig, axs = plt.subplots(n_split)
        for i in range(n_split):
            axs[i].plot(history[history_ixs].history['accuracy'], label='accuracy')
            axs[i].plot(history[history_ixs].history['val_accuracy'], label='val_accuracy')
            axs[i].set_title(f'Fold {i}')
            history_ixs += 1
    else:
        fig, axs = plt.subplots(train_size, n_split)
        for i in range(train_size):
            for j in range(n_split):
                axs[i, j].plot(history[history_ixs].history['accuracy'], label='accuracy')
                axs[i, j].plot(history[history_ixs].history['val_accuracy'], label='val_accuracy')
                axs[i, j].set_title(f'Fold {j} - batch {i}')
                history_ixs += 1
    for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel='Accuracy', ylim=[0,1])
    plt.show()

history_multiplot(history, train_ds_size, 5)
#test_loss, test_acc = model.evaluate(val_ds, verbose=2)
##test_loss, test_acc = model_an.evaluate(val_ds, verbose=2)
#
#print(test_acc)
quit()
indices = [0, 2, 4, 6] #encontrar indices de capas convolucionales que vienen antes de las de pooling #0 3 6 8 10
outputs = [model.layers[i].output for i in indices]
model1 = models.Model(inputs=model.inputs, outputs=outputs)
#cargar imagen
img = load_img('./NuevaData/No_sanos/Captura CARIES  SEVERISIMMA.PNG', target_size=(150,150))
#convert the image to an array
img = img_to_array(img)
#expand dimensions so that it represents a single 'sample'
img = np.expand_dims(img, axis=0)
#get feature map for first hidden layer
feature_maps = model1.predict(img)
#plot the output from each block 
square = 4 
for fmap in feature_maps:
    #plot all 16 maps in an 4x4 squares
    ix = 1
    for _ in range(square):
        for _ in range(square):
            #specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            #plot filter channel in grayscale
            plt.imshow(fmap[0, :, :, ix-1])
            ix += 1
    #show the figure
    plt.show()
