from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when
import apache_beam as beam
import pandas as pd
from kmodes.kprototypes import KPrototypes

spark = SparkSession.builder.master(f"local")\
          .appName("AdultCSVData")\
          .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
path = "Source Data/adult.csv"
processed_path = "Processed_data/"
renamed_cols = ['age','workclass','final_weight','education','education_num',
                'marital_status','occupation','relationship','race','sex','capital_gain',
                'capital_loss','hours_per_week','native_country','income']

class load_data(beam.DoFn):
    def process(self,element):
        path = element

        df = spark.read.csv(path,inferSchema=True,header=True)
        print("\nThe data is loaded\n")
        df.printSchema()
        
        return [df.toPandas()]
    
class rename_columns(beam.DoFn):

    def process(self,df,columns):

        print("\nRenaming column\n")

        df = spark.createDataFrame(df)
        df = df.toDF(*columns)
        df.printSchema()
        
        return [df.toPandas()]
    
class split_data(beam.DoFn):

    def stratified_split(self,data,ratios=[0.8,0.2],target_col='income'):
        pos_sample = data.filter(col(target_col)=='>50K')
        neg_sample = data.filter(col(target_col)=='<=50K')

        train_pos, test_pos = pos_sample.randomSplit(ratios, seed=42)
        train_neg, test_neg = neg_sample.randomSplit(ratios, seed=42)

        train = train_pos.union(train_neg)
        test = test_pos.union(test_neg)

        return train.toPandas(), test.toPandas()
    
    def process(self,data):

        data = spark.createDataFrame(data)

        train, test = self.stratified_split(data)

        yield (train,test)
    
"""class check(beam.DoFn):
    def process(self,element):
        print("\nTesting\n")
        T1=element[0]
        T2=element[1]
        print(T1.head())
        print(T2.head())
    """

class missing_valueImpute(beam.DoFn):

    def fillMissingValues(self,data,num_clusters=5):
        pandas_data = data.toPandas()
        target = pandas_data.pop('income')

        missing_cols = pandas_data.columns[pandas_data.isnull().sum()>0]
        print("Columns with missing values : ",missing_cols.tolist())

        missing_data_cols = pandas_data[missing_cols]
        pandas_data.drop(missing_cols,axis=1,inplace=True)

        object_columns = pandas_data.select_dtypes(include=['object']).columns
        object_column_indices = [pandas_data.columns.get_loc(col) for col in object_columns]
        print("Categorical columns with no missing values : ",object_column_indices)

        kproto = KPrototypes(n_clusters=num_clusters, init='Cao', verbose=1, n_init=1)

        clusters = kproto.fit_predict(pandas_data, categorical = object_column_indices)

        pandas_data['Cluster'] = clusters

        pandas_data[missing_cols] = missing_data_cols

        for cluster_id in range(num_clusters):
            cluster_data = pandas_data[pandas_data['Cluster'] == cluster_id]

            cluster_modes = cluster_data.mode().iloc[0]

            for col in missing_cols:
                pandas_data.loc[pandas_data['Cluster'] == cluster_id,col] = pandas_data.loc[pandas_data['Cluster'] == cluster_id, col].fillna(cluster_modes[col])
            
        pandas_data.drop('Cluster',axis=1,inplace=True)

        pandas_data['income'] = target
        
        #spark_data = spark.createDataFrame(pandas_data)

        return pandas_data



    def process(self, element):
        train = element[0]
        test = element[1]

        train = spark.createDataFrame(train)
        test = spark.createDataFrame(test)

        train = train.replace('?',None)
        test = test.replace('?',None)

        null_counts_train = [(column, train.filter(col(column).isNull()).count()) for column in train.columns]
        null_counts_test = [(column, test.filter(col(column).isNull()).count()) for column in test.columns]

        print("\nNull values in train\n")
        print(null_counts_train)
        print("\nNull values in test\n")
        print(null_counts_test)

        print("\nImputing missing values in Train\n")
        Imputed_train = self.fillMissingValues(train)
        print("\nImputing missing values in Test\n")
        Imputed_test = self.fillMissingValues(test)

        Imputed_train.to_csv(processed_path+"Imputed_train.csv",index=False)

        Imputed_test.to_csv(processed_path+"Imputed_test.csv",index=False)



with beam.Pipeline() as p:
    sparkDataframe = (p | 'load Data' >> beam.Create([path]) 
                       | beam.ParDo(load_data()))
        
    (sparkDataframe |'rename columns'>>beam.ParDo(rename_columns(),renamed_cols)
                    |'split the data'>>beam.ParDo(split_data())
                    |'check'>>beam.ParDo(missing_valueImpute()))
    

spark.stop()





