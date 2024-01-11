from scripts.data_related import preprocess

import numpy as np
import pandas as pd

from tensorflow import Tensor
from keras.layers import Input, ReLU, BatchNormalization, Dense, Softmax, Dropout
from keras.regularizers import L1L2
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def load_data(annotations_for_files):

    X, Y = preprocess.get_annotated_csi_segments_from_esp_logs(annotations_for_files, 1, True)

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
        'Shape of Training Data:\nX={}, Y={}\n\nShape of Validation Data:\nX={}, Y={}\n\nShape of Test Data:\nX={}, Y={}\n\n'.format(
            X_train.shape, Y_train.shape,
            X_val.shape, Y_val.shape,
            X_test.shape, Y_test.shape
        )
    )

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def dense_bn_relu_drop(x: Tensor, num_neuron, dropout_rate=0.0, l1l2_regularizer=None) -> Tensor:

    x = Dense(num_neuron, kernel_regularizer=l1l2_regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_rate)(x)

    return x


def create_fc_model(input_shape, num_classes, num_neuron_list, dropout_rate=0.0):

    l1l2_regularizer = L1L2(l1=1e-5, l2=1e-4)

    inputs = Input(shape=input_shape)
    t = BatchNormalization()(inputs)

    for num_neuron in num_neuron_list:
        t = dense_bn_relu_drop(t, num_neuron, dropout_rate, l1l2_regularizer)

    t = Dense(num_classes, kernel_regularizer=l1l2_regularizer)(t)
    t = BatchNormalization()(t)
    outputs = Softmax()(t)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    return model


def main(annotations_for_files, model_formats, application, save_model_folder_path, save_result_file_path):

    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(annotations_for_files)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        verbose=1,
        mode="min",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=5,
    )
    accuracies = []
    model_names = []

    for model_format in model_formats:
        model = create_fc_model(
            input_shape=X_train[0].shape,
            num_classes=Y_train.shape[1],
            num_neuron_list=model_format,
            dropout_rate=0.2
        )

        model.fit(
            x=X_train,
            y=Y_train,
            batch_size=16,
            epochs=100,
            verbose=1,
            validation_data=(X_val, Y_val),
            callbacks=[early_stopping]
        )

        model_name = '{}_fc_{}'.format(
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
            '../data/human_activity_recognition/corridor_standing_1685750223.3215554.txt']
    }
    '''
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
    '''

    model_formats = [
        [],
        [25], [50],
        [50, 50], [100, 100],
        [50, 100, 100, 50], [100, 150, 150, 100], [150, 200, 200, 150],
        [100, 175, 250, 250, 175, 100]
    ]

    main(
        annotations_for_files,
        model_formats,
        'pd',
        '../models/keras/har_final/fc_single_run_3',
        '../data/har_final/pd_fc_single_run_3.csv'
    )
