from scripts.data_related import preprocess
import tinyml_opt as opt

import tensorflow as tf
import numpy as np
import pandas as pd
import pathlib
import os.path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def load_data(annotations_for_files, flatten):

    X, Y = preprocess.get_annotated_csi_segments_from_esp_logs(annotations_for_files, 50, flatten)
    X = preprocess.get_amplitudes(X)
    Y_categorical = to_categorical(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_categorical, train_size=0.90, shuffle=True, random_state=1)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.90, shuffle=True, random_state=1)

    print(
        'Shape of Training Data:\nX={}, Y={}\n\nShape of Validation Data:\nX={}, Y={}\n\nShape of Test Data:\nX={}, Y={}'.format(
            X_train.shape, Y_train.shape,
            X_val.shape, Y_val.shape,
            X_test.shape, Y_test.shape
        )
    )

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def save_tflite_model(tflite_model, tflite_save_path):
    tflite_model_filepath = pathlib.Path(tflite_save_path)
    bytes_written = tflite_model_filepath.write_bytes(tflite_model)
    print('For TFLite Model: {}, KBs written are: {}'.format(tflite_save_path, bytes_written / 1024))

    return bytes_written / 1024


def infer_tflite_model(tflite_save_path, X_test, Y_test):

    interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    Y_test_hat_lite = []
    X_test = (X_test/input_details[0]['quantization_parameters']['scales'][0]) + input_details[0]['quantization_parameters']['zero_points'][0]
    X_test_lite = X_test.astype(input_details[0]['dtype'])
    for x in X_test_lite:
        interpreter.set_tensor(input_details[0]['index'], np.reshape(x, input_details[0]['shape']))
        interpreter.invoke()
        quant_op = interpreter.get_tensor(output_details[0]['index'])[0]
        quant_op = quant_op.astype(np.float32)
        quant_op = (quant_op - (output_details[0]['quantization_parameters']['zero_points'][0])) * (output_details[0]['quantization_parameters']['scales'][0])
        Y_test_hat_lite.append(quant_op)

    Y_test_hat_lite = np.reshape(np.array(Y_test_hat_lite), Y_test.shape)
    Y_test_hat_lite = to_categorical(np.argmax(Y_test_hat_lite, axis=1), num_classes=Y_test.shape[1])
    accuracy = accuracy_score(Y_test, Y_test_hat_lite)
    print('For TFLite Model: {}, Accuracy is: {}'.format(tflite_save_path, accuracy))

    return accuracy


def save_run_statistics(file_path, model_names, optimizations, tflite_model_sizes, tflite_model_accuracies):

    data_dict = {
        'model': np.repeat(model_names, len(optimizations)),
        'optimizations': optimizations * len(model_names),
        'size_kb': tflite_model_sizes,
        'accuracy': tflite_model_accuracies
    }

    data_df = pd.DataFrame(data_dict)
    data_df.to_csv(file_path)

    print('TFLite Stats saved to path: {}'.format(file_path))


def apply_optimization(base_model, opt_type, X_train, Y_train, X_val, Y_val, epochs, model_folder, model_name):

    if opt_type == 'noopt':
        return opt.no_opt(
            base_model
        )

    elif opt_type == 'pruning':
        return opt.pruning(
            base_model,
            X_train, Y_train, X_val, Y_val,
            epochs,
            model_folder, model_name
        )

    elif opt_type == 'clustering':
        return opt.clustering(
            base_model,
            X_train, Y_train, X_val, Y_val,
            epochs,
            model_folder, model_name
        )

    elif opt_type == 'qat':
        return opt.quantization_aware_training(
            base_model,
            X_train, Y_train, X_val, Y_val,
            epochs,
            model_folder, model_name
        )

    elif opt_type == 'ptq':
        return opt.post_training_quantization(
            base_model,
            X_val
        )

    elif opt_type == 'cqat':
        return opt.cluster_preserving_qat(
            base_model,
            X_train, Y_train, X_val, Y_val,
            True,
            epochs,
            model_folder, model_name
        )

    elif opt_type == 'pqat':
        return opt.sparsity_preserving_qat(
            base_model,
            X_train, Y_train, X_val, Y_val,
            True,
            epochs,
            model_folder, model_name
        )

    elif opt_type == 'cptq':
        return opt.cluster_preserving_qat(
            base_model,
            X_train, Y_train, X_val, Y_val,
            False,
            epochs,
            model_folder, model_name
        )

    elif opt_type == 'pptq':
        return opt.sparsity_preserving_qat(
            base_model,
            X_train, Y_train, X_val, Y_val,
            False,
            epochs,
            model_folder, model_name
        )

    elif opt_type == 'pruning_clustering':
        return opt.sparsity_preserving_clustering(
            base_model,
            X_train, Y_train, X_val, Y_val,
            epochs,
            model_folder, model_name
        )

    elif opt_type == 'pcqat':
        return opt.cluster_and_sparsity_preserving_qat(
            base_model,
            X_train, Y_train, X_val, Y_val,
            True,
            epochs,
            model_folder, model_name
        )

    elif opt_type == 'pcptq':
        return opt.cluster_and_sparsity_preserving_qat(
            base_model,
            X_train, Y_train, X_val, Y_val,
            False,
            epochs,
            model_folder, model_name
        )

    else:
        raise Exception('{}, no such optimization technique found'.format(opt_type))


def main(identifier, flatten):

    annotations_for_files = {
        0: ['../data/human_activity_recognition/lab_empty_1685504223.572277.txt',
            '../data/human_activity_recognition/parking_empty_1685740580.4567273.txt',
            '../data/human_activity_recognition/meetingroom_empty_1685749485.6155138.txt'],
        1: ['../data/human_activity_recognition/lab_standing_1685504666.5088024.txt',
            '../data/human_activity_recognition/parking_standing_1685741178.8062696.txt',
            '../data/human_activity_recognition/meetingroom_standing_1685750223.3215554.txt'],
        2: ['../data/human_activity_recognition/lab_sitting_1685505160.02395.txt',
            '../data/human_activity_recognition/parking_sitting_1685741578.0545967.txt',
            '../data/human_activity_recognition/meetingroom_sitting_1685751375.4694774.txt'],
        3: ['../data/human_activity_recognition/lab_sittingupdown_1685506366.902033.txt',
            '../data/human_activity_recognition/parking_sittingupdown_1685741945.5226183.txt',
            '../data/human_activity_recognition/meetingroom_sittingupdown_1685751017.898958.txt'],
        4: ['../data/human_activity_recognition/lab_jumping_1685508067.7758608.txt',
            '../data/human_activity_recognition/parking_jumping_1685743199.2157178.txt',
            '../data/human_activity_recognition/meetingroom_jumping_1685751761.0926085.txt'],
        5: ['../data/human_activity_recognition/lab_walking_fast_1685507226.876064.txt',
            '../data/human_activity_recognition/parking_walking_fast_1685742761.396288.txt',
            '../data/human_activity_recognition/meetingroom_walking_fast_1685750603.6436868.txt']
    }

    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(annotations_for_files, flatten)

    keras_model_folder = '../models/keras/har_final/{}'.format(identifier)
    tflite_model_folder = '../models/tflite/har_final/{}'.format(identifier)
    model_names = [
        'har_resnet_',
        'har_resnet_1', 'har_resnet_2',
        'har_resnet_11', 'har_resnet_22',
        'har_resnet_111', 'har_resnet_222',
        'har_resnet_1111', 'har_resnet_2222'
    ]
    optimizations = [
        'ptq'
    ]
    tflite_model_sizes = []
    tflite_model_accuracies = []

    for model_name in model_names:
        for optimization in optimizations:

            base_model = tf.keras.models.load_model('{}/{}'.format(keras_model_folder, model_name))
            tflite_model_save_path = '{}/{}_{}.tflite'.format(tflite_model_folder, model_name, optimization)

            if os.path.isfile(tflite_model_save_path):
                tflite_model_size = os.path.getsize(tflite_model_save_path)/1024

            else:
                try:
                    tflite_model = apply_optimization(
                        base_model, optimization,
                        X_train, Y_train, X_val, Y_val,
                        2,  # Epochs for fine-tuning
                        keras_model_folder, model_name
                    )
                except Exception as err:
                    print(err.args)
                    return

                tflite_model_size = save_tflite_model(tflite_model, tflite_model_save_path)

            tflite_model_accuracy = infer_tflite_model(tflite_model_save_path, X_test, Y_test)

            tflite_model_sizes.append(tflite_model_size)
            tflite_model_accuracies.append(tflite_model_accuracy)

    save_run_statistics(
        '../data/tflite_stats/{}_optimizations.csv'.format(identifier),
        model_names, optimizations,
        tflite_model_sizes, tflite_model_accuracies
    )


if __name__ == '__main__':
    identifiers = ['cnn_full_int_run_3']
    for identifier in identifiers:
        main(identifier, False)



