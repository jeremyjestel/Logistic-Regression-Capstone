import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score

def predict(features):
    data = pd.read_csv('clean_dataset.csv')
    X = data.drop('Approved', axis=1)
    y = data['Approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    undersampler = RandomUnderSampler(random_state=42)
    X_train, y_train = undersampler.fit_resample(X_train, y_train)

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    return model.predict(features)