## Scripts

We divide the codebase into three parts as described below.
The scripts to each part are stored in `data_related`, `train_models` and `compress_models` respectively:

#### Data Prepossessing and Visualization
Here we have scripts that are used to read the ESP32 log files in the `data/human_activity_recognition` directory and compute the CSI amplitude spectrograms that are used for training the models.
We have also provided a sample iPython notebook, `data_viz.ipynb`, that can be used to plot and visualize the spectrograms.

#### Building and Training Models
This folder contains code that we use to generate models with different architecture and parameter sizes.
It also trains the model on the data given in `data/human_activity_recognition` folder and tests it, giving us a accuracy measure.

#### Compressing Models
These contains code that are used to compress the saved KERAS models (in `models/keras`) and convert them to TF-LITE MODEL (later saved in `models\tflite`).
The main file is `tf_lite_convert.py` which uses TinyML optimization functions defined in file `tinyml_opt.py`, along with some utility functions defined in `tf_lite_convert_utils.py`.
It also contains script in file `convert_tflite_to_cc.py` to convert all tflite models in a folder to C++ header files used by the sensing applications running in ESP32.

The details of what every function in a script does is described in the comments of that script file.