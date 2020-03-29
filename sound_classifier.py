import numpy as np
from numpy import genfromtxt
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import pandas as pd

# extracting data from csv files into numpy arrays
x_train = genfromtxt('train_data.csv', delimiter=',')
y_train = genfromtxt('train_labels.csv', delimiter=',')
x_test = genfromtxt('test_data.csv', delimiter=',')
y_test = genfromtxt('test_labels.csv', delimiter=',')
print("#1 data extraction done")

# shape
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# converting to one hot
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print(y_train.shape, y_test.shape)
print("#2 conversion done")

# reshaping to 2D
x_train = np.reshape(x_train, (x_train.shape[0], 40, 5))
x_test = np.reshape(x_test, (x_test.shape[0], 40, 5))
print(x_train.shape, x_test.shape)
print("#3 2d reshape done")

# reshaping to shape required by CNN
x_train = np.reshape(x_train, (x_train.shape[0], 40, 5, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 40, 5, 1))
print("#4 reshape for CNN done")

# shapes
print(x_train.shape, x_test.shape)

# forming model
model = Sequential()

# adding layers and forming the model
model.add(Conv2D(64, kernel_size=5, strides=1, padding="Same", activation="relu", input_shape=(40, 5, 1)))
model.add(MaxPooling2D(padding="same"))
model.add(Conv2D(128, kernel_size=5, strides=1, padding="same", activation="relu"))
model.add(MaxPooling2D(padding="same"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation="softmax"))
print("#5 model forming done")

# compiling
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("#6 compilation done")

# training the model
model.fit(x_train, y_train, batch_size=50, epochs=30, validation_data=(x_test, y_test))
print("#7 training done")

# train and test loss and scores respectively
train_loss_score = model.evaluate(x_train, y_train)
test_loss_score = model.evaluate(x_test, y_test)
print(train_loss_score)
print(test_loss_score)

classes = ["air_conditioner", "car_horn", "children_playing",
           "dog_bark", "drilling", "engine_idling",
           "gun_shot", "jackhammer", "siren", "street_music"]
predicted_classes = model.predict_classes(x_test)
predicted_explanations = []
for x in predicted_classes:
    predicted_explanations.append(classes[x])

# ~~~~~~~~~~~~~~~~~  activations & heatmaps  ~~~~~~~~~~~~~~~~~
from keract import get_activations, display_activations, display_heatmaps

for i in range(837):
    eval_item = x_test[i:i + 1]
    activations = get_activations(model, eval_item, "conv2d_1")
    # display_activations(activations, save=False)
    display_heatmaps(activations, eval_item, "./heatmaps/" + str(predicted_explanations[i]) + "/heatmap" + str(i),
                     save=True)
    print("#" + str(i) + ": ", predicted_explanations[i])
    # TODO interpret heatmap?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data = {"SoundID": list(range(1, len(predicted_classes) + 1)), "Label": predicted_classes,
        "Explanation": predicted_explanations}
submissions = pd.DataFrame(data)
submissions.to_csv("submission2.csv", index=False, header=True)

print("DONE")

# results
# [0.061496452033782796, 0.9820139408111572]
# [1.0961064827765294, 0.7335723042488098]

# new
# [0.10706214858587784, 0.9680810570716858]
# [0.9786004304058022, 0.7574671506881714]
