from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import os
 
CATEGORY = [ 'correct_mask', 'incorrect_mask', 'without_mask' ]

# this is a np.array() that each element is a
train_images = []
train_labels = []
validation_images = []
validation_labels = []
test_images = []
test_labels = []
label = 0

for category in CATEGORY:
    path = os.path.join("dataset", category)
    listdir = os.listdir(path)
    random.shuffle(listdir)
    
    _70_percent_train = int(len(listdir) * 0.7)
    # 15 phần trăm validation, còn lại sẽ là test
    _15_percent_validation = int(len(listdir) - _70_percent_train) // 2

    print(f"{category} size: {len(listdir)}")
    print(f"train size: {_70_percent_train}")
    print(f"validation size: {_15_percent_validation}")
    print(f"test size: {len(listdir) - _70_percent_train - _15_percent_validation}\n")
    
    # training data
    first = 0
    last = _70_percent_train
    for img in listdir[first : last]:
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        train_images.append(image)
        train_labels.append(label)
    
    # validation data
    first = last
    last = first + _15_percent_validation
    for img in listdir[first : last]:
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        validation_images.append(image)
        validation_labels.append(label)

    # test data
    first = last
    for img in listdir[first : -1]:
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        test_images.append(image)
        test_labels.append(label)
    label += 1

train_images = np.array(train_images, dtype='float32')
train_labels = np.array(train_labels)
validation_images = np.array(validation_images, dtype='float32')
validation_labels = np.array(validation_labels)
test_images = np.array(test_images, dtype='float32')
test_labels = np.array(test_labels)

# conv layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))

# fc layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(validation_images, validation_labels),
                    batch_size=32)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig("plot_cnn.png")

pred = model.predict(test_images, batch_size=32)

# in ra thống kê kết quả của tập test
print(classification_report(test_labels, pred.argmax(axis=1), 
                            target_names=['correct_mask',  'incorrect_mask', 'without_mask']))

test_loss, test_acc = model.evaluate(test_images,  test_labels)


print("test loss: ", test_loss)
print("test accuracy: ", test_acc)

model.save("mask_detector.model", save_format="h5")
