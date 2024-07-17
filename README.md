<h1> Apache Spark and Beam demo </h1>
This repository shows how to scale data processing and Model deployment using two famous frameworks Spark and Beam in Python. For the demonstration purpose the dataset is kept small, but various data preprocessing techniques such as missing value imputation, SMOTE have been performed. Three base classifiers from Spark ML-Lib were used. Because pyspark library lacks stack ensemble classifier we used custom stack classifier to ensemble the base learners. 
<h3>List of files</h3>
<ul>
  <li><strong>Problem Statement</strong>: contains the detailed procedure to be followed. Although we  made some small changes in it.</li>
  <li><strong>Source Data</strong>: contains original data without any preprocessing.</li>
  <li><strong>Processed Data</strong>: contains the data post processing like missing value imputation.</li>
  <li><strong>train.csv</strong>: contains the data for train set.</li>
  <li><strong>test.csv</strong>: contains the data for test set.</li>
  <li><strong>Data_preprocessor.py</strong>: the main source code for the data processing.</li>
  <li><strong>ML_model_training.py</strong>: the main source code for the training pipeline.</li>
  <li><strong>jupyter_notebook_version.ipynb</strong>: same code as above two but in jupyter notebook.</li>
</ul>
