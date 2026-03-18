### LIBRARIES ### 
# Data
import pandas as pd

# Math
import numpy as np

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns

### GLOBAL SETUP ### 
GITHUB_PATH = "https://raw.githubusercontent.com/RafaSFernandes/data-cleaning-and-wrangling-airbnb/refs/heads/main/data/"

### MAIN FUNCTION ###

def process(filename1, filename2, df_overlayed=None):
  """
    Process data from files given filenames.
  """
  df = read_data(filename1, filename2)

  basic_info(df)

  missing_values = df.isnull().sum()
  missing_values = missing_values[missing_values > 0]

  duplicated_values = df.duplicated()
  duplicated_values = duplicated_values[duplicated_values > 0]

  if missing_values.any:
    null_hand(df)
  
  if duplicated_values.any:
    duplicated_hand(df)
  
  cols_types(df)
  df = outliers(df)

  df = transform_cat_cols(df)
  df = normalize(df)

  display(df.dtypes)
  
  boxplots(df, df_overlayed)

  return df

### LOADING FUNCTIONS ###

def read_data(filename1, filename2):
  """
    Read and merge data from folder given filenames.
  """
  df = pd.merge(pd.read_csv(GITHUB_PATH + filename1), pd.read_csv(GITHUB_PATH + filename2), how='left', on='id')
  return df

def basic_info(df):
  """
    Display basic information about the dataframe.
  """
  print("Dimensions (Rows, Columns)", df.shape)
  missing_values = df.isnull().sum()
  missing_values = missing_values[missing_values > 0]
  duplicated_values = df.duplicated()
  duplicated_values = duplicated_values[duplicated_values > 0]
  if missing_values.empty:
    print('No missing values.')
  else:
    print('There are missing values.')
    print(f"Missing values columns\n{missing_values.index}")

  if duplicated_values.empty:
    print('No duplicated values.')
  else:
    print('There are duplicated values.')
    print(f"Duplicated rows\n{duplicated_values.index}")

  return

### CLEANING AND WRANGLING FUNCIONS ###

def null_hand(df):
  """
    Handle null values.
  """
  num_cols = df.select_dtypes(include='number').drop('id', axis=1).columns
  df[num_cols] = df[num_cols].fillna(df[num_cols].median())
  return df

def duplicated_hand(df):
  """
    Handle duplicated values.
  """
  df = df.drop_duplicates()
  return df

def cols_types(df):
  """
    Change columns types.
  """
  int_cols = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'number_of_reviews']
  df[int_cols] = df[int_cols].astype(int)
  
  return df[int_cols]

def outliers(df):
  """
    Remove outliers from dataframe.
  """
  num_cols = df.select_dtypes(include='number').drop('id', axis=1).columns
  outliers_cols = []

  for col in num_cols: 
    Q1 = df[col].quantile(0.25) 
    Q3 = df[col].quantile(0.75) 
    IQR = Q3 - Q1 
    LI = Q1 - 1.5*IQR 
    LS = Q3 + 1.5*IQR 

    df['outliers_' + col] = ((df[col] < LI) | (df[col] > LS))
    outliers_cols.append('outliers_' + col)

  for col in outliers_cols:
    df = df[df[col] == False]
    df = df.drop(columns=col)

  return df

### TRANSFORMATION FUNCTIONS ###

def transform_cat_cols(df):
  """
    Transform categorical columns.
  """
  label_encode_cols = ['neighbourhood_cleansed']
  onehot_encode_cols = ['room_type']

  for col in label_encode_cols:
    df = label_encode_col(df, col).drop(col, axis=1)

  for col in onehot_encode_cols:
    df = onehot_encode_col(df, col)
  return df

def label_encode_col(df, column_name):
  """
    Label encode column.
  """
  df[column_name + '_number'] = df[column_name].astype('category').cat.codes
  return df

def onehot_encode_col(df, column_name):
  """
    One-hot encode column.
  """
  df_dummies = pd.get_dummies(df[column_name], dtype=int, prefix='room_type')
  df = pd.concat([df.drop(columns=column_name, axis=1), df_dummies], axis=1)
  return df

### NORMALIZE FUNCTIONS ###
def normalize(df):
  """
    Normalize dataframe.
  """
  columns = df.select_dtypes(include='number').drop('id', axis=1).columns
  for col in columns:
    df = normalize_column(df, col)
  return df

def normalize_column(df, column_name):
  """
    Normalize column.
  """
  df[column_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
  return df

### GRAPHICS FUNCTIONS ###

def boxplots(df, df_overlayed=None):
  """
    Plot boxplots.
  """
  plt.figure(figsize=(20, 6))

  sns.boxplot(data=df.drop('id', axis=1), color="red", boxprops=dict(alpha=0.5))

  # 'df_overlayed' boxplot
  if df_overlayed is not None:
    sns.boxplot(data=df_overlayed.drop('id', axis=1), color="yellow", boxprops=dict(alpha=0.5))

  plt.xticks(rotation=30, ha='right')
  plt.title("Overlayed Boxplots (Red: This Data Frame | Yellow: Main Data Frame) | Overlay result: Orange")
  plt.show()
  
  return
