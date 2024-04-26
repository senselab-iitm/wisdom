## On-Device Deep Learning for IoT-based Wireless Sensing Applications

[<a href='https://drive.google.com/drive/u/2/folders/1QzZPx9LiYrjlb1CArLRRkUAeOjM13ahV'>Datasets</a>] [<a href='https://drive.google.com/drive/u/2/folders/1Tzo6rNLU8OlriODip4Zmgpos7e0sL9c9'>Models</a>] [<a href='./papers/workshop_wisdom.pdf'>Workshop Paper</a>] [<a href='./papers/artifact_wisdom.pdf'>Artifact Paper</a>] [<a href='http://cse.iitm.ac.in/~sense/wisdom'>Project Website</a>]

The GitHub repository only consists of the scripts. 
The dataset and models are available publicly on <a href='https://drive.google.com/drive/u/2/folders/13Crp-owAzkjZVH85AhisW9Yfi78wsoMf'>Google Drive</a>.
To know more about our work please read the <a href=''>workshop paper (TBA) </a> or check our <a href='http://cse.iitm.ac.in/~sense/wisdom'>project website</a>. 
To know more about the organisation of the project and how to use the datasets, models and script, please have a look at the <a href='./papers/artifact_wisdom.pdf'>artifact paper</a> or the readme files present in the different sub-folders of the project.

Once you download the data and models put them inside the main `wisdom` folder i.e., wisdom will have three folder `data`, `models`, `scripts`, this will ensure the filepaths/imports in the code to work properly.

The high level directory structure should look like below. 
Please go through additional **readme** files for more information:

* wisdom
  * data
    * current_measurements
      * non_compressed
      * quantized
      * **curr_mea_readme.md**
    * human_activity_recognition
      * indoors
      * outdoors
      * **esp_col_desp.md**
      * **har_readme.md**
  * models
    * keras
    * tflite
    * **models_readme.md**
  * scripts
    * compress_models
    * data_related
    * train_models
    * **scripts_readme.md**
