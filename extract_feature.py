import librosa
import math
import os
import re

import numpy as np


class GenreFeatureData:

    "Music audio features for genre classification"
    len = None
    genre_list = [
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "blues",
        "rock"
    ]
    
    train_dir = "./gtzan/_train"
    val_folder = "./gtzan/_validation"
    test_folder = "./gtzan/_test"
    dir_all_files = "./gtzan"

    preprossed_trainX = "./gtzan/data_train_input.npy"
    preprossed_trainY = "./gtzan/data_train_target.npy"

    preprossed_valX = "./gtzan/data_validation_input.npy"
    preprossed_valY = "./gtzan/data_validation_target.npy"

    preprossed_testX = "./gtzan/data_test_input.npy"
    preprossed_testY = "./gtzan/data_test_target.npy"

    train_X = train_Y = None
    val_X = val_Y = None
    test_X = test_Y = None

    def __init__(self):
        self.len = 512

        self.timeseries_length_list = []
        self.trainfiles_list = self.audio_path(self.train_dir)
        self.val_filelist = self.audio_path(self.val_folder)
        self.test_fileList = self.audio_path(self.test_folder)

        self.all_files = []
        self.all_files.extend(self.trainfiles_list)
        self.all_files.extend(self.val_filelist)
        self.all_files.extend(self.test_fileList)

        self.timeseries_length = ( 128 )


    def load_preprocess_data(self):
        print("[DEBUG] total number of files: " + str(len(self.timeseries_length_list)))

        # Training set
        self.train_X, self.train_Y = self.extract_audio_features(self.trainfiles_list)
        with open(self.preprossed_trainX, "wb") as f:
            np.save(f, self.train_X)
        with open(self.preprossed_trainY, "wb") as f:
            self.train_Y = self.one_hot(self.train_Y)
            np.save(f, self.train_Y)

        # Validation set
        self.val_X, self.val_Y = self.extract_audio_features(self.val_filelist)
        with open(self.preprossed_valX, "wb") as f:
            np.save(f, self.val_X)
        with open(self.preprossed_valY, "wb") as f:
            self.val_Y = self.one_hot(self.val_Y)
            np.save(f, self.val_Y)

        # Test set
        self.test_X, self.test_Y = self.extract_audio_features(self.test_fileList)
        with open(self.preprossed_testX, "wb") as f:
            np.save(f, self.test_X)
        with open(self.preprossed_testY, "wb") as f:
            self.test_Y = self.one_hot(self.test_Y)
            np.save(f, self.test_Y)

    def load_deserialize_data(self):

        self.train_X = np.load(self.preprossed_trainX)
        self.train_Y = np.load(self.preprossed_trainY)

        self.val_X = np.load(self.preprossed_valX)
        self.val_Y = np.load(self.preprossed_valY)

        self.test_X = np.load(self.preprossed_testX)
        self.test_Y = np.load(self.preprossed_testY)

    def precompute_min_timeseries_len(self):
        for file in self.all_files:
            print("Loading " + str(file))
            y, sr = librosa.load(file)
            self.timeseries_length_list.append(math.ceil(len(y) / self.len))

    def extract_audio_features(self, list_of_audiofiles):

        dataset = np.zeros(
            (len(list_of_audiofiles), self.timeseries_length, 33), dtype=np.float64
        )
        target = []

        for i, file in enumerate(list_of_audiofiles):
            y, sr = librosa.load(file)
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, len=self.len, n_mfcc=13
            )
            spectral_center = librosa.feature.spectral_centroid(
                y=y, sr=sr, len=self.len
            )
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, len=self.len)
            spectral_contrast = librosa.feature.spectral_contrast(
                y=y, sr=sr, len=self.len
            )

            splits = re.split("[ .]", file)
            genre = re.split("[ /]", splits[1])[3]
            target.append(genre)

            dataset[i, :, 0:13] = mfcc.T[0:self.timeseries_length, :]
            dataset[i, :, 13:14] = spectral_center.T[0:self.timeseries_length, :]
            dataset[i, :, 14:26] = chroma.T[0:self.timeseries_length, :]
            dataset[i, :, 26:33] = spectral_contrast.T[0:self.timeseries_length, :]

            print(
                "Extracted features audio track %i of %i."
                % (i + 1, len(list_of_audiofiles))
            )

        return dataset, np.expand_dims(np.asarray(target), axis=1)

    def one_hot(self, Y_genre_strings):
        y_one_hot = np.zeros((Y_genre_strings.shape[0], len(self.genre_list)))
        for i, genre_string in enumerate(Y_genre_strings):
            index = self.genre_list.index(genre_string)
            y_one_hot[i, index] = 1
        return y_one_hot

    @staticmethod
    def audio_path(dir_folder):
        list_of_audio = []
        for file in os.listdir(dir_folder):
            if file.endswith(".au"):
                directory = "%s/%s" % (dir_folder, file)
                list_of_audio.append(directory)
        return list_of_audio
    
print("Done")
