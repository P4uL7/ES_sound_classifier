import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

# forming a panda dataframe from the metadata file
data = pd.read_csv("/home/paul/Desktop/UrbanSound8K/metadata/UrbanSound8K.csv")
print("#1 dataframe done")

# head of the dataframe
print(data.head())

# count of datapoints in each of the folders
print(data["fold"].value_counts())

# preprocessing using entire feature set
x_train = []
x_test = []
y_train = []
y_test = []
path = "/home/paul/Desktop/UrbanSound8K/audio/fold"

for i in tqdm(range(len(data))):
    fold_no = str(data.iloc[i]["fold"])
    file = data.iloc[i]["slice_file_name"]
    label = data.iloc[i]["classID"]
    filename = path + fold_no + "/" + file
    y, sr = librosa.load(filename)
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T, axis=0)
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T, axis=0)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40).T, axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40).T, axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40).T, axis=0)
    features = np.reshape(np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)), (40, 5))
    if fold_no != '10':
        x_train.append(features)
        y_train.append(label)
    else:
        x_test.append(features)
        y_test.append(label)
print(len(x_train) + len(x_test), len(data))
print("#2 preprocessing done")

# converting the lists into numpy arrays
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print("#3 numpy conversion done")

# reshaping into 2d to save in csv format
x_train_2d = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test_2d = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
print(x_train_2d.shape, x_test_2d.shape)
print("#4 2d reshape done")

# saving the data numpy arrays
np.savetxt("train_data.csv", x_train_2d, delimiter=",")
np.savetxt("test_data.csv", x_test_2d, delimiter=",")
np.savetxt("train_labels.csv", y_train, delimiter=",")
np.savetxt("test_labels.csv", y_test, delimiter=",")
print("#5 numpy arrays done")
