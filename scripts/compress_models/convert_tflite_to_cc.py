import os


'''
:param
    - model_folder: path to the folder w.r.t the script file. C++ files are generated for all the TFLite models in model_folder
'''
def convert_tflite_to_cc(model_folder):

    all_models = os.listdir(model_folder)
    converted_models = [x.replace('tflite', 'cc') for x in all_models]
    for model, converted_model in zip(all_models, converted_models):
        if not os.path.isfile('{}/{}'.format(model_folder, converted_model)):
            os.system('xxd -i {}/{} > {}/{}'.format(model_folder, model, model_folder, converted_model))


if __name__ == '__main__':
    model_folder = '../models/tflite/cnn/cnn_run1'
    convert_tflite_to_cc(model_folder)

