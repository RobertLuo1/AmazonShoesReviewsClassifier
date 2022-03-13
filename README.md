## The Framework Of The Project
### Explanation of files and directories

- #### blendingData
  -  blending_X_new.npy
     
     - The blending data for the classifier

  - blending_y_new.npy
    - The blending label for the classifier
  - y_predict_randomForest.npy
    - The prediction from RandomForest classifier
  - y_predict_svc.npy
    - The prediction from SVC classifier
  - y_true_randomForest.npy
    - The true label
  - y_true_svc.npy
    - The true label

- #### Model

  - Bilstm_model60 29.54.pth
    - The checkpoint of the model(weights)
  - svc.pkl
    - SVC model for blending model
  - ws1.pkl
    - The dictionary for transforming the word to sequence

- #### original_data

  - Different types of  reviews crawled from websites 

- #### picture

  - All the pictures for the project including the evaluation of tf-idf and neural network and loss
  - The length 15-50.png is for finding the best input length.
  - accuracy-Top_num_relationship.png for finding the best top k for tf-idf.
  
- #### tools

  - reviews_union.csv
    - All the reviews collectively crawled from Amazon websites
  - tf_idf_table.csv
    - Each tf-idf value for each word of document

- #### Bilstm.py

  - The neural network model construction

- #### blending_model_test.py

  - The evaluation of the blending model

- #### blending.py

  - Get the data predicted by two models on training data

- #### config.py

  - The configurations for all the project

- #### crawler.py

  - The crawler for crawling the reviews from Amazon websites about shoes item

- #### data_preparation.py

  - Pre-processing the data mentioned in report

- #### evaluate.py

  - The evaluation of the neural network model

- #### main.py

  - In order to make the dictionary available for the all files, this file helps to generate the dictionary.

- #### Prediction_tf_idf.py

  - This file is for testing the generalization of th-idf model

- #### preparation_tf_idf.py

  - It helps to generate the tf-idf value for each word of each document.

- #### test_tf_idf.py

  - It is the evaluation of tf-idf clssifier

- #### textdataset

  - Preparation for the following input for neural network
  
- #### Train_process

  - The training process of neural network

- #### UI

  - The file for user to test their reviews to see whether it is positive or negative

- #### wordtosequence

  - encode the word into number for the embedding layer



### How to use it

- #### packages for this project

  - ##### numpy

  - ##### pandas

  - ##### nltk

  - ##### sklearn

  - ##### torch

  - ##### tqdm

  - ##### matplotlib

  - ##### selenium

- #### Train

  Directly run the code **python Train_process.py** and if you want to change the hyper parameters, please check it out on the **config.py**.

- #### Evaluate

  Generally, there are three models in the project and if you want to run the evaluation of tf-idf model, simply run the code **python test_tf_idf.py** and you can find the metrics.

  If you want to check out the evaluation of neural network, please run the code **python evaluate.py**

  For the blending model, you can run the **python blending_model_test.py** for the results.

- #### UI

  The UI is for the user who wants to use this system, if you want to try our system and you just need to run the code **python UI.py** and following the instruction and you can get the results.

### Demo

![image-20220313222528985](https://gitee.com/luo-zhuoyan-58119327/myimage/raw/master/img/image-20220313222528985.png)

