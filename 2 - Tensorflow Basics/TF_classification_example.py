'''Classification example using TensorFlow framework'''
'''Topics: tf.estimator API, categorical and continuous features, LinearClassifier, DNNClassifier'''

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import tensorflow as tf

diabetes = pd.read_csv('pima-indians-diabetes.csv')
# print(diabetes.head())
# print(diabetes.columns)
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure',
                'Triceps', 'Insulin', 'BMI', 'Pedigree']
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (
    x - x.min()) / x.max() - x.min())  # normalize data in numerical columns

print(tf.__version__)

# # dataframe to features convertion
# # continuous features
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.featurefeature_column_columns.numeric_column('Pedigree')
# can be treated as a continuous column
age = tf.feature_column.numeric_column('Age')
#
# # categorical columns
# assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A', 'B', 'C'.'D'])
# # assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group',hash_bucket_size=10)
#
# # Feature engineering, domain knowledge -> convert Age from continuous feature to categorical
# age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70, 80])

