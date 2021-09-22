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
# learning rate
INIT_LR = 1e-4
# số lần lướt qua epoch
EPOCHS = 20
# batch size = 20
BS = 32

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
'''
Các thuộc tính trong thư mục dataset.
Đây cũng là các nhãn mà mô hình sẽ dự đoán.
'''
CATEGORY = [ 'incorrect_mask', 'correct_mask', 'without_mask' ]

'''
Đầu ra của mô hình một vector dạng onehot, với 3 kết quả như sau:
[1, 0, 0]: correct
[0, 1, 0]: incorrect
[0, 0, 1]: without
'''

# this is a np.array() that each element is a
train_images = []
train_labels = []
validation_images = []    # validation
validation_labels = []    # validation
test_images = []
test_labels = []
# thếm 2 mảng test
# thiếu test

for category in CATEGORY:
    # tạo đường dẫn: dataset/"category"
    # với category là 1 trong 3 phần tử của mảng CATEGORY (dòng 40)
    path = os.path.join("dataset", category)
    # lấy danh sách các file trong đường dẫn path
    listdir = os.listdir(path)
    # xáo trộn danh sách vừa lấy được ở trên.
    random.shuffle(listdir)
    # 70 phần trăm số lượng ảnh trong thư mục path
    _70_percent_train = int(len(listdir) * 0.7)
    # 15 phần trăm validation, còn lại sẽ là test
    _15_percent_validation = int(len(listdir) - _70_percent_train) // 2
    
    # in các thông tin về số lượng ảnh trong thư mục và số lượng ảnh sử dụng cho train, validation, và test
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
        preprocess_input(image)
        train_images.append(image)
        train_labels.append(category)
    
    # validation data
    first = last
    last = first + _15_percent_validation
    for img in listdir[first : last]:
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        preprocess_input(image)
        validation_images.append(image)
        validation_labels.append(category)

    # test data
    first = last
    for img in listdir[first : -1]:
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        preprocess_input(image)
        test_images.append(image)
        test_labels.append(category)

train_images = np.array(train_images, dtype='float32')
train_labels = np.array(train_labels)
validation_images = np.array(validation_images, dtype='float32')
validation_labels = np.array(validation_labels)
test_images = np.array(test_images, dtype='float32')
test_labels = np.array(test_labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
validation_labels = lb.fit_transform(validation_labels)
test_labels = lb.fit_transform(test_labels)

# construct the training image generator for data augmentation
# tìm hiêu các đối số này
'''
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
'''
aug = ImageDataGenerator(
	rotation_range=40,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# sử dụng MobileNetV2 với fc bị loại bỏ
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# tùy biến fc để gắn vào basemodel
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
# tìm hiểu dòng này
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

# gắn headmodel vào basemodel để tạo ra mô hình hoàn chỉnh
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# biên dịch model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# huấn luyện fc
print("[INFO] training head...")
H = model.fit(
	aug.flow(train_images, train_labels, batch_size=BS),
	steps_per_epoch=len(train_images) // BS,
	validation_data=(validation_images, validation_labels),
	validation_steps=len(validation_images) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(test_images, batch_size=BS)

# mội ảnh sẽ được mô hình đoán ra 3 xác suất tương ứng với 3 nhãn
# tạo ra một mảng mà mỗi phần tử là index của nhãn có xác suất cao nhất.
predIdxs = np.argmax(predIdxs, axis=1)

# in ra thống kê kết quả của tập test
print(classification_report(test_labels.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# lưu model vào file và lưu trên đĩa cứng
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# vẽ biểu đồ loss và accuracy của của quá trình train.
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
