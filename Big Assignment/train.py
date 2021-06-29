from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# initialize the initial learning rate, number of epochs to train for,
# and batch size
'''
INIT_LR = 1e-4
EPOCHS = 20
BS = 32
'''
# chỉnh thử
INIT_LR = 1e-4
# xuonsg 20 cũng được
EPOCHS = 25
# batch siz: thử nhé
BS = 32

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
CATEGORY = [ 'incorrect_mask', 'correct_mask', 'without_mask' ]

DIRECTORY = 'dataset/'
'''
đầu ra là onehot
[1, 0, 0]: correct
[0, 1, 0]: incorrect
[0, 0, 1]: without
'''

# this is a np.array() that each element is a
train_images = []
train_labels = []
test_images = []    # validation
test_labels = []    # validation
# thếm 2 mảng test
# thiếu test

for category in CATEGORY:
    path = os.path.join(DIRECTORY, category)
    
    listdir = os.listdir(path)
    
    random.shuffle(listdir)

    _70_percent = int(len(listdir) * 0.7)
    
    print(f"{category}: {len(listdir)}")
    print(f"train size: {_70_percent}")
    print(f"test_size: {len(listdir) - _70_percent}\n")
    
    # training data
    for img in listdir[0:_70_percent]:
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        preprocess_input(image)
        train_images.append(image)
        train_labels.append(category)
    
    # testing data
    for img in listdir[_70_percent:-1]:
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        preprocess_input(image)
        test_images.append(image)
        test_labels.append(category)

train_images = np.array(train_images, dtype='float32')
train_labels = np.array(train_labels)
test_images = np.array(test_images, dtype='float32')
test_labels = np.array(test_labels)

'''
print(train_images.shape)
print(train_labels[0])
print(test_images.shape)
print(test_labels[0])
'''

# perform one-hot encoding on the labels
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.fit_transform(test_labels)

# construct the training image generator for data augmentation
# tìm hiêu các đối số này
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load a MobileNetV2 model with fully connected layer left off
# fully connected layer will be customize later
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct a fully connected layer which will be put to head of model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
# tìm hiểu dòng này
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

'''
print(train_images.shape)
print(train_labels[0])
print(test_images.shape)
print(test_labels[0])
'''

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(train_images, train_labels, batch_size=BS),
	steps_per_epoch=len(train_images) // BS,
	validation_data=(test_images, test_labels),
	validation_steps=len(test_images) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(test_images, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(test_labels.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch nth")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
