from scripts.data_related import preprocess

import numpy as np
import pandas as pd

from tensorflow import Tensor
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, GlobalAvgPool2D, Dense
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
    - inputs: an input tensorflow Tensor
    
:returns
    - bn: pass the input Tensor through a layer of ReLU non-linearity and then batch normalization
'''
def relu_bn(inputs: Tensor) -> Tensor:

    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)

    return bn


'''
:params
    - x: input Tensor
    - downsample: If true the output tensor size is halved, else it remains the same
    - filters: number of filters for each convolutional layer
    - kernel_size: the size of each filter, for e.g., 3 means a 3X3 filter
    
:returns
    - out: Tensor x, after it passes through 2 covnolutional layer with one ReLU and batch normalization layer in between.
    This is then added back with x (skip connection). In case we are down sampling the skip connection also has a convolutional layer to match the dimensions.
'''
def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:

    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)

    out = Add()([x, y])
    out = relu_bn(out)

    return out

'''
:params
    - input_shape: shape of the input to the model (number of packets X subcarriers)
    - initial_num_filters: initial number of filters. Filters increases as the number of convolutional blocks increases
    - num_classes: number of classes for classification task. For HAR there are 6 classes
    - num_blocks_list: list of number. Each number is the number of blocks after which number of filters double.
    For e.g. [1, 2, 1] double filter after one block, then double filter after two blocks, then finally double filter after the last block
    
:returns
    - model: The final resnet-like model. The final model consists of a global average pooling with softmax for output.
'''
def create_res_net(input_shape, initial_num_filters, num_classes, num_blocks_list):

    inputs = Input(shape=input_shape)
    num_filters = initial_num_filters

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)

    for num_blocks in num_blocks_list:
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0), filters=num_filters)
        num_filters *= 2

    t = GlobalAvgPool2D()(t)
    outputs = Dense(num_classes, activation='softmax')(t)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

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

        model = create_res_net(
            input_shape=(X_train.shape[1], X_train.shape[2], 1),
            initial_num_filters=8,
            num_classes=Y_train.shape[1],
            num_blocks_list=model_format
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

        model_name = '{}_resnet_{}'.format(
            application,
            str(model_format).replace('[', '').replace(']', '').replace(', ', '')
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
        0: ['../../data/human_activity_recognition/indoors/lab/lab_empty_1685504223.572277.txt',
            '../../data/human_activity_recognition/outdoors/parking/parking_empty_1685740580.4567273.txt',
            '../../data/human_activity_recognition/indoors/corridor/corridor_empty_1685749485.6155138.txt'],
        1: ['../../data/human_activity_recognition/indoors/lab/lab_standing_1685504666.5088024.txt',
            '../../data/human_activity_recognition/outdoors/parking/parking_standing_1685741178.8062696.txt',
            '../../data/human_activity_recognition/indoors/corridor/corridor_standing_1685750223.3215554.txt'],
        2: ['../../data/human_activity_recognition/indoors/lab/lab_sitting_1685505160.02395.txt',
            '../../data/human_activity_recognition/outdoors/parking/parking_sitting_1685741578.0545967.txt',
            '../../data/human_activity_recognition/indoors/corridor/corridor_sitting_1685751375.4694774.txt'],
        3: ['../../data/human_activity_recognition/indoors/lab/lab_sittingupdown_1685506366.902033.txt',
            '../../data/human_activity_recognition/outdoors/parking/parking_sittingupdown_1685741945.5226183.txt',
            '../../data/human_activity_recognition/indoors/corridor/corridor_sittingupdown_1685751017.898958.txt'],
        4: ['../../data/human_activity_recognition/indoors/lab/lab_jumping_1685508067.7758608.txt',
            '../../data/human_activity_recognition/outdoors/parking/parking_jumping_1685743199.2157178.txt',
            '../../data/human_activity_recognition/indoors/corridor/corridor_jumping_1685751761.0926085.txt'],
        5: ['../../data/human_activity_recognition/indoors/lab/lab_walking_1685507226.876064.txt',
            '../../data/human_activity_recognition/outdoors/parking/parking_walking_1685742761.396288.txt',
            '../../data/human_activity_recognition/indoors/corridor/corridor_walking_1685750603.6436868.txt']
    }

    model_formats = [
        [],
        [1],
        [2],
        [1, 1],
        [2, 2],
        [1, 1, 1],
        [2, 2, 2],
        [1, 1, 1, 1],
        [2, 2, 2, 2]
    ]

    main(
        annotations_for_files,
        model_formats,
        'har',
        '../../models/keras/cnn/cnn_run4',
        '../../data/train_results/cnn_run4.csv'
    )
