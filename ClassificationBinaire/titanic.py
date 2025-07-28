import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, default_collate

def preprocess_titanic_data():
  """
  Basic processing of the Titanic dataset for usage within ML Training
  """
  df = pd.read_csv('./titanic/train.csv')
  df.head()
  df = df.drop(columns=['Name', 'Ticket', 'Cabin'], axis=1)
  df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
  df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
  df['Age'] = df['Age'].fillna(df['Age'].median())
  df = df.dropna()
  labels = df.pop('Survived')
  df.head()
  # Split data into train and test/eval
  X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2)
  return X_train, X_test, y_train, y_test


class TitanicDataset(Dataset):
    def __init__(self, samples, labels):
        self.df = samples
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.tensor(self.df.iloc[idx].values, dtype=torch.float32)
        y = torch.tensor(self.labels.iloc[idx], dtype=torch.float32)
        return x, y

def get_titanic_data_as_dataset():
    X_train, X_test, y_train, y_test = preprocess_titanic_data()
    train_dataset = TitanicDataset(X_train, y_train)
    eval_dataset = TitanicDataset(X_test, y_test)
    return train_dataset, eval_dataset


def solve_with_randomforest():
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score
  X_train, X_test, y_train, y_test = preprocess_titanic_data()
  rf = RandomForestClassifier(n_estimators=100, random_state=42)
  rf.fit(X_train, y_train)
  y_pred = rf.predict(X_test)
  print(f"RandomForestClassifier: {accuracy_score(y_test, y_pred)=}")

if __name__ == '__main__':
  solve_with_randomforest()