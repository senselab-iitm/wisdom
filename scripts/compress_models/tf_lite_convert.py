import sys
sys.path.insert(1, '../data_related')
import preprocess

import tinyml_opt as opt

import tensorflow as tf
import numpy as np
import pandas as pd
import pathlib
import os.path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


'''
:params
    - annotations_for_files: list of annotations, where each annotation has a list of file associated to it
    - flatten: boolean value to decide it the spectograms that will be used as input needs to be flattened (a single M*N vector) or not (a MXN matrix)
    
:returns
    - X_train, X_val, X_test, Y_train, Y_val, Y_test: Training, validation and testing datasets
'''
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


'''
:params
    - tflite_model: the tflite model to be saved
    - tflite_save_path: the path where tflite_model will be saved

:returns
    - kb_written: the size of tflite_model in KBs
'''
def save_tflite_model(tflite_model, tflite_save_path):
    tflite_model_filepath = pathlib.Path(tflite_save_path)
    bytes_written = tflite_model_filepath.write_bytes(tflite_model)
    print('For TFLite Model: {}, KBs written are: {}'.format(tflite_save_path, bytes_written / 1024))
    kb_written = bytes_written / 1024
    return kb_written


'''
:params
    - tflite_save_path: the path of the saved tflite model on which we want to perform test
    - X_test, Y_test: the test data and labels to be used for testing
    
:returns
    - accuracy: the accuracy of the model saved in tflite_save_path when inferred using X_test, Y_test dataset
'''
def infer_tflite_model(tflite_save_path, X_test, Y_test):

    interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    Y_test_hat_lite = []
    X_test_lite = X_test.astype(input_details[0]['dtype'])
    for x in X_test_lite:
        interpreter.set_tensor(input_details[0]['index'], np.reshape(x, input_details[0]['shape']))
        interpreter.invoke()
        quant_op = interpreter.get_tensor(output_details[0]['index'])[0]
        Y_test_hat_lite.append(quant_op)

    Y_test_hat_lite = np.reshape(np.array(Y_test_hat_lite), Y_test.shape)
    Y_test_hat_lite = to_categorical(np.argmax(Y_test_hat_lite, axis=1), num_classes=Y_test.shape[1])
    accuracy = accuracy_score(Y_test, Y_test_hat_lite)
    print('For TFLite Model: {}, Accuracy is: {}'.format(tflite_save_path, accuracy))

    return accuracy


'''
:params
    - file_path: path where we want to save the statistics (accuracy and memory) or a particular run (creating, training and compressing all possible combinations)
    - models_names: names of the models used in the run, in the format of har_<architecture_type>_<config>. Here <config> determines the parameter size.
    - optimizations: names of optimization techniques used in the run
    - tflite_model_sizes: size of all the models used in the run. This is different from flash consumed.
    - tflite_model_accuracies: size of all the models used in the run, found using infer_tflite_model
'''
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


'''
:params
    - base_model: base Keras model on which compression needs to be performed
    - opt_type: type of model compression to be performed on base_model
    - X_train, Y_train, X_val, Y_val: Used for fine turning the model when further trained during compression
    - epochs: number of additional epochs for fine tuning the model for compression
    - model_folder: path of the folder where model should be saved
    - model_name: name of the tflite model to be saved
:returns
    - tflite model: returns the tflite model generated after compressing base_model. It is also saved in model_folder/model_name file
'''
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


'''
:params
    - identifier: name of the run. Each run has all the combination of architecture, parameters and compression techniques
    - flatten: whether the CSI spectrogram needs to be converted to a 1D vector or they remain a 2D matrix
'''
def main(identifier, flatten):

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

    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(annotations_for_files, flatten)

    keras_model_folder = '../../models/keras/cnn/{}'.format(identifier)
    tflite_model_folder = '../../models/tflite/cnn/{}'.format(identifier)
    model_names = [
        'har_resnet_',
        'har_resnet_1', 'har_resnet_2',
        'har_resnet_11', 'har_resnet_22',
        'har_resnet_111', 'har_resnet_222',
        'har_resnet_1111', 'har_resnet_2222'
    ]
    optimizations = [
        'noopt',
        'pruning', 'clustering', 'qat', 'ptq',
        'cqat', 'cptq', 'pqat', 'pptq', 'pruning_clustering',
        'pcqat', 'pcptq'
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
        '../../data/compression_results/{}.csv'.format(identifier),
        model_names, optimizations,
        tflite_model_sizes, tflite_model_accuracies
    )


if __name__ == '__main__':
    identifiers = ['cnn_run4']
    for identifier in identifiers:
        main(identifier, False)




