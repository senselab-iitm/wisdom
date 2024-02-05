from scripts.data_related import preprocess

import numpy as np
import pandas as pd

from keras.layers import Input, BatchNormalization, Dense, LSTM
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

'''
:params
    - annotations_for_files: list of annotations, where each annotation has a list of file associated to it

:returns
    - X_train, X_val, X_test, Y_train, Y_val, Y_test: Training, validation and testing datasets
'''
def load_data(annotations_for_files):

    X, Y = preprocess.get_annotated_csi_segments_from_esp_logs(annotations_for_files, 50, False)

    X = abs(X).astype(np.float32)
    max = np.percentile(X, 99)
    X[np.nonzero(X > max)] = max

    Y_categorical = to_categorical(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_categorical,
        train_size=0.90, shuffle=True, random_state=1
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train,
        train_size=0.90, shuffle=True, random_state=1
    )

    print(
        'Shape of Training Data:\nX={}, Y={}\n\nShape of Validation Data:\nX={}, Y={}\n\nShape of Test Data:\nX={}, Y={}'.format(
            X_train.shape, Y_train.shape,
            X_val.shape, Y_val.shape,
            X_test.shape, Y_test.shape
        )
    )

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


'''
:params
    - input_shape: shape of the input to the model (number of packets X subcarriers)
    - num_classes: number of classes for classification task. For HAR there are 6 classes
    - units: list of number. Each number is the dimensionality of the output space for a LSTM cell

:returns
    - model: The final LSTM model
'''
def create_rnn(input_shape, num_classes, units):
    inputs = Input(shape=input_shape)
    t = BatchNormalization()(inputs)

    for unit in units[:-1]:
        t = LSTM(units=unit, return_sequences=True)(t)

    t = LSTM(units=units[-1], return_sequences=False)(t)
    outputs = Dense(num_classes, activation='softmax')(t)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


'''
:params
    - annotations_for_files: list of annotations, where each annotation corresponds to a list of files that have the esp logs
    - model_formats: list of CNN model formats i.e., a list of num_blocks_list for create_res_net
    - application: application name i.e., har in our case. Used for naming the models
    - save_model_folder_path: path to the folder where the model will be saved
    - save_result_file_path: path to the file where we save the results i.e., model name with the accuracy it achieved
'''
def main(annotations_for_files, model_formats, application, save_model_folder_path, save_result_file_path):

    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(annotations_for_files)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=10,
        verbose=1,
        mode="min",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=5,
    )
    accuracies = []
    model_names = []

    for model_format in model_formats:

        model = create_rnn(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=Y_train.shape[1],
            units=model_format
        )

        model.fit(
            x=X_train,
            y=Y_train,
            batch_size=8,
            epochs=100,
            verbose=1,
            validation_data=(X_val, Y_val),
            callbacks=[early_stopping]
        )

        model_name = '{}_rnn_{}'.format(
            application,
            str(model_format).replace('[', '').replace(']', '').replace(', ', '_')
        )
        _, accuracy = model.evaluate(X_test, Y_test, verbose=0)
        print('Accuracy for {} model is {}'.format(model_name, accuracy))
        accuracies.append(accuracy)
        model_names.append(model_name)

        export_dir = '{}/{}'.format(
            save_model_folder_path,
            model_name
        )
        model.save(export_dir)

    df = pd.DataFrame({
        'model': model_names,
        'accuracy': accuracies
    })
    df.to_csv(save_result_file_path)


if __name__ == '__main__':

    annotations_for_files = {
        0: ['../data/human_activity_recognition/lab_empty_1685504223.572277.txt',
            '../data/human_activity_recognition/parking_empty_1685740580.4567273.txt',
            '../data/human_activity_recognition/corridor_empty_1685749485.6155138.txt'],
        1: ['../data/human_activity_recognition/lab_standing_1685504666.5088024.txt',
            '../data/human_activity_recognition/parking_standing_1685741178.8062696.txt',
            '../data/human_activity_recognition/corridor_standing_1685750223.3215554.txt'],
        2: ['../data/human_activity_recognition/lab_sitting_1685505160.02395.txt',
            '../data/human_activity_recognition/parking_sitting_1685741578.0545967.txt',
            '../data/human_activity_recognition/corridor_sitting_1685751375.4694774.txt'],
        3: ['../data/human_activity_recognition/lab_sittingupdown_1685506366.902033.txt',
            '../data/human_activity_recognition/parking_sittingupdown_1685741945.5226183.txt',
            '../data/human_activity_recognition/corridor_sittingupdown_1685751017.898958.txt'],
        4: ['../data/human_activity_recognition/lab_jumping_1685508067.7758608.txt',
            '../data/human_activity_recognition/parking_jumping_1685743199.2157178.txt',
            '../data/human_activity_recognition/corridor_jumping_1685751761.0926085.txt'],
        5: ['../data/human_activity_recognition/lab_walking_fast_1685507226.876064.txt',
            '../data/human_activity_recognition/parking_walking_fast_1685742761.396288.txt',
            '../data/human_activity_recognition/meetingroom_walking_fast_1685750603.6436868.txt']
    }

    model_formats = [
        [2], [7], [10], [18], [34], [50],
        [70, 35],
        [90, 45, 20], [140, 70, 35]
    ]

    main(
        annotations_for_files,
        model_formats,
        'har',
        '../../models/keras/rnn/rnn_run1',
        '../data/har_final/rnn_run1.csv'
    )
