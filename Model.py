import pandas as pd
#import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

# fetch dataset
def eye_opener():
    df = pd.read_csv('EEG_Eye_State_Classification.csv')

    # data (as pandas dataframes)
    X = df.iloc[:, :-1]  # all columns except the last
    y = df['eyeDetection']  # last column (eye state)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    parameters = {'n_estimators':[5, 10, 50, 100, 500],
                  #'criterion':('gini', 'entropy', 'log loss'),
                  #'max_depth': [2,3,4,5],
                  #'min_samples_split':[20,30,40],
                  #'min_samples_leaf':[10,20,30]
                  }

    random_forest = RandomForestClassifier(random_state=42)

    grid_search =  GridSearchCV(estimator=random_forest,param_grid=parameters, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

    grid_search.fit(X_train,y_train)

    print("Best hyperparameters", grid_search.best_params_)
    print("Best accuracy on training data:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    test_predicted = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predicted)
    print("Test Accuracy:", test_accuracy)


if __name__ == "__main__":
    eye_opener()