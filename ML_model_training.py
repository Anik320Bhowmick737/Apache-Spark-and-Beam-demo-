from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when
import apache_beam as beam
import pandas as pd
from pyspark.ml.feature import StringIndexer, OneHotEncoder, MinMaxScaler, VectorAssembler, StandardScaler
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from imblearn.over_sampling import SMOTE
from pyspark.ml.functions import vector_to_array
import numpy as np
import os

spark = SparkSession.builder.master(f"local")\
          .appName("AdultCSVData")\
          .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

AU_ROC_calc = BinaryClassificationEvaluator(metricName='areaUnderROC',labelCol='label')
Accuracy_calc = MulticlassClassificationEvaluator(metricName='accuracy',labelCol='label')
F1_calc = MulticlassClassificationEvaluator(metricName='f1')

class StackClassifier:
    def __init__(self,base_learners):
        self.base_learners = base_learners
        self.assembler = None

    def extract_probabs(self,dataframe,name):
        return dataframe.select(vector_to_array("probability")[1].alias("probs_"+str(name))).toPandas()

    def intermediate_transform(self,model,i,data):
        return self.extract_probabs(model.transform(data),i)

    def get_sparkdata(self,combined_pandas_list,label_col):
        combined_pandas_data = pd.concat(combined_pandas_list,axis=1)
        feature_cols = combined_pandas_data.columns.values
        combined_pandas_data['label'] = label_col

        spark_dataframe = spark.createDataFrame(combined_pandas_data)

        if self.assembler==None:
            self.assembler = VectorAssembler(inputCols = feature_cols ,outputCol="features")
            transformed = self.assembler.transform(spark_dataframe)
        else:
            transformed = self.assembler.transform(spark_dataframe)
        return transformed


    def fit(self, data):
        Label = data.select('label').toPandas()
        output_pandas_data = []
        self.trained_baselearners = []
        print("Training base models")
        for i,bs in enumerate(self.base_learners):
            print("Training model number: ",i+1)
            model = bs.fit(data)
            output_pandas_data.append(self.intermediate_transform(model,i,data))
            #print(Accuracy_calc.evaluate(model.transform(data)))
            self.trained_baselearners.append(model)

        final_data = self.get_sparkdata(output_pandas_data,Label)

        self.gbtClassifier = GBTClassifier()
        print("Training the meta classifier")
        self.gbt_model = self.gbtClassifier.fit(final_data)

    def transform(self,data):
        Label = data.select('label').toPandas()
        output_pandas_data = []
        for i,bs in enumerate(self.trained_baselearners):
            #print(Accuracy_calc.evaluate(bs.transform(data)))
            output_pandas_data.append(self.intermediate_transform(bs,i,data))

        final_data = self.get_sparkdata(output_pandas_data,Label)

        return self.gbt_model.transform(final_data)

source_data_path = "Processed_data"

class load_data(beam.DoFn):
    def process(self,element):
        print("\nLoading the data\n")
        files = os.listdir(element)
        train_name = files[0]
        #print(f'\n{train_name}\n')
        test_name =files[1]
        #print(f'\n{test_name}\n')

        train_data = pd.read_csv(os.path.join(source_data_path,train_name))
        test_data = pd.read_csv(os.path.join(source_data_path,test_name))

        return [(train_data, test_data)]
    
class prepare_data(beam.DoFn):
    def process(self,element):
        
        print("\nPreparing the data\n")

        train = spark.createDataFrame(element[0])
        test = spark.createDataFrame(element[1])

        categorical_cols = []
        index_categorical_cols = []
        ohe_categorical_cols = []

        numerical_cols = []

        for colum, dtype in train.dtypes:  
            if dtype=='string':
                categorical_cols.append(colum)
                index_categorical_cols.append(colum+"Index")
                ohe_categorical_cols.append(colum+"_ohv")
            else:
                numerical_cols.append(colum)

        indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_categorical_cols)# Apply only string indexer to the income no One Hot encoding
        encoder = OneHotEncoder(inputCols = indexer.getOutputCols()[:-1], outputCols = ohe_categorical_cols[:-1])
        vector_assembler_1 = VectorAssembler(inputCols = numerical_cols, outputCol = 'InputFeatures')
        scaler = MinMaxScaler(inputCol = 'InputFeatures', outputCol = "ScaledFeatures")
        vector_assembler_2 = VectorAssembler(inputCols = ["ScaledFeatures"] + encoder.getOutputCols(), outputCol = 'features')

        pipe = Pipeline(stages=[indexer,encoder,vector_assembler_1,scaler, vector_assembler_2])

        model_pipe = pipe.fit(train)

        train = model_pipe.transform(train).select('features','incomeIndex')

        print("Viewing the final processed train data")
        train.show()
        test = model_pipe.transform(test).select('features','incomeIndex').withColumnRenamed('incomeIndex','label')

        X = test.select('features').rdd.map(lambda row: DenseVector(row[0]).toArray()).collect()
        Y = np.array(test.select('label').rdd.map(lambda row: row[0]).collect())

        test = map(lambda x,y: (DenseVector(x),int(y)),X,Y)

        test = spark.createDataFrame(test, schema = ['features','label'])

        return [(train.toPandas(),test.toPandas())]
    
class perform_SMOTE(beam.DoFn):
    def process(self,element):

        print("\nPerforming SMOTE on train data\n")

        train = spark.createDataFrame(element)
        #test = element[1]

        X_train = train.select('features').rdd.map(lambda row: DenseVector(row[0]).toArray()).collect()
        y_train = np.array(train.select('incomeIndex').rdd.map(lambda row: row[0]).collect())

        sm = SMOTE(random_state = 42)
        X_res,y_res = sm.fit_resample(X_train,y_train)

        df_train = map(lambda x,y: (DenseVector(x),int(y)),X_res,y_res)

        df_train = spark.createDataFrame(df_train, schema = ['features','label'])

        return [df_train.toPandas()]
    

class train_BaseLearners(beam.DoFn):

    def print_metrics(self, predicted_train,predicted_test):
        print("\nTraining metrics\n")

        print(f"Train accuracy: {Accuracy_calc.evaluate(predicted_train):.4f}")
        print(f"Train AU-ROC: {AU_ROC_calc.evaluate(predicted_train):.4f}")
        print(f"Train F1: {F1_calc.evaluate(predicted_train):.4f}")

        print("\nTesting metrics\n")

        print(f"Train accuracy: {Accuracy_calc.evaluate(predicted_test):.4f}")
        print(f"Train AU-ROC: {AU_ROC_calc.evaluate(predicted_test):.4f}")
        print(f"Train F1: {F1_calc.evaluate(predicted_test):.4f}")


    def process(self,element):

        df_train = spark.createDataFrame(element[0])
        df_test = spark.createDataFrame(element[1])
        
        print("\nTraining Base learners\n")
        print("\nTraining Logistic regression\n")

        base_LR = LogisticRegression(featuresCol = 'features', labelCol = 'label',
                             aggregationDepth = 2,
                             elasticNetParam = 0.0,
                             regParam = 0.0,
                             standardization = True,
                             threshold =  0.5,
                             tol =  1e-06,
                             maxIter = 100)
        
        lr_model = base_LR.fit(df_train)
        predicted_train = lr_model.transform(df_train)

        predicted_test = lr_model.transform(df_test)

        self.print_metrics(predicted_train,predicted_test)

        print("\nTraining DecisionTree classificier\n")

        base_DT = DecisionTreeClassifier(featuresCol='features', labelCol='label',
                                 checkpointInterval = 10, impurity = 'entropy',
                                 maxBins = 128, maxDepth = 15, minInstancesPerNode = 1 )
        
        dt_model = base_DT.fit(df_train)
        predicted_train = dt_model.transform(df_train)

        predicted_test = dt_model.transform(df_test)

        self.print_metrics(predicted_train,predicted_test)

        print("\nTraining RandomForest classificier\n")

        base_RF = RandomForestClassifier(featuresCol = 'features', labelCol = 'label',
                                 featureSubsetStrategy = 'sqrt', impurity = 'gini', subsamplingRate = 0.8,
                                 numTrees = 50, maxDepth = 15)
        
        rf_model = base_RF.fit(df_train)
        predicted_train = rf_model.transform(df_train)

        predicted_test = rf_model.transform(df_test)

        self.print_metrics(predicted_train,predicted_test)




class Train_ensemble(beam.DoFn):

    def print_metrics(self, predicted_train,predicted_test):
        
        print("\nTraining metrics\n")

        print(f"Train accuracy: {Accuracy_calc.evaluate(predicted_train):.4f}")
        print(f"Train AU-ROC: {AU_ROC_calc.evaluate(predicted_train):.4f}")
        print(f"Train F1: {F1_calc.evaluate(predicted_train):.4f}")

        print("\nTesting metrics\n")

        print(f"Train accuracy: {Accuracy_calc.evaluate(predicted_test):.4f}")
        print(f"Train AU-ROC: {AU_ROC_calc.evaluate(predicted_test):.4f}")
        print(f"Train F1: {F1_calc.evaluate(predicted_test):.4f}")

    def process(self,element):

        df_train = spark.createDataFrame(element[0])
        df_test = spark.createDataFrame(element[1])

        base_LR = LogisticRegression(featuresCol = 'features', labelCol = 'label',
                             aggregationDepth = 2,
                             elasticNetParam = 0.0,
                             regParam = 0.0,
                             standardization = True,
                             threshold =  0.5,
                             tol =  1e-06,
                             maxIter = 100)

        base_DT = DecisionTreeClassifier(featuresCol='features', labelCol='label',
                                 checkpointInterval = 10, impurity = 'entropy',
                                 maxBins = 128, maxDepth = 15, minInstancesPerNode = 1 )

        base_RF = RandomForestClassifier(featuresCol = 'features', labelCol = 'label',
                                 featureSubsetStrategy = 'sqrt', impurity = 'gini', subsamplingRate = 0.8,
                                 numTrees = 50, maxDepth = 15)
        
        St = StackClassifier([base_LR,base_DT,base_RF])

        print("\nTraining the ensembles of model\n")
        St.fit(df_train)
        train_predicted = St.transform(df_train)
        test_predicted = St.transform(df_test)
        
        self.print_metrics(train_predicted,test_predicted)


def combine(df_1,df_2):
    print("Combning the data")
    print(df_1.head())
    print(df_2.head())
    return [(df_1,df_2)]




with beam.Pipeline() as p:
    DataFrames = (p | 'load Data' >> beam.Create([source_data_path]) 
                    | beam.ParDo(load_data()))
    
        
    processed_data = DataFrames |'prepare the data'>>beam.ParDo(prepare_data())

    train_data = processed_data|beam.Map(lambda x:x[0])
    test_data = processed_data|beam.Map(lambda x:x[1])

    transformed_data = train_data | 'perform SMOTE' >> beam.ParDo(perform_SMOTE())
    
    #combined_data = transformed_data|beam.Map(combine,beam.pvalue.AsSingleton(test_data))
    combined_data = transformed_data|beam.Map(lambda x,test: (x,test),beam.pvalue.AsSingleton(test_data))

    combined_data | beam.ParDo(train_BaseLearners())
    combined_data | beam.ParDo(Train_ensemble())


    #let's save the data

    combined_data | beam.Map(lambda x: x[0].to_csv("train_data.csv", index=False))
    combined_data | beam.Map(lambda x: x[1].to_csv("test_data.csv", index=False))

spark.stop()


    
   




                    
                    

        







        












            








