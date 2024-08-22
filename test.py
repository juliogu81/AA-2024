import pandas as pd
from sklearn import preprocessing, model_selection
DATASET_FILE = 'lab1_dataset.csv'

dataset = pd.read_csv(DATASET_FILE, sep=",", header=None).add_prefix("c")
print(
    f"{dataset.shape[0]} records read from {DATASET_FILE}\n{dataset.shape[1]} attributes found"
)
dataset.head(10)

dataset.c24.value_counts()

train, test = model_selection.train_test_split(dataset, test_size=0.2, random_state=42)
print(f"{train.shape[0]} samples for training, {test.shape[0]} samples for testing")
train.head(10)