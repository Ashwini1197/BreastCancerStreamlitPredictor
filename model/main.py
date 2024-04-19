import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle5 as pickle


def create_model(data): 
  X = data.drop(['diagnosis'], axis=1)
  y = data['diagnosis']
  
  # scale the data
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  # split the data
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )
  
  # train the model
  model = LogisticRegression()
  model.fit(X_train, y_train)
  
  # test model
  y_pred = model.predict(X_test)
  print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
  print("Classification report: \n", classification_report(y_test, y_pred))
  
  return model, scaler

def compare_models(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Define models and parameters
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier()
    }
    params = {
        'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 10]},
        'Random Forest': {'n_estimators': [10, 50, 100], 'max_features': ['auto', 'sqrt', 'log2']},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'KNN': {'n_neighbors': [3, 5, 7, 9], 'metric': ['euclidean', 'manhattan']}
    }
    
    best_models = {}
    for name, model in models.items():
        clf = GridSearchCV(model, params[name], cv=5, scoring='accuracy')
        clf.fit(X_train, y_train)
        best_models[name] = (clf.best_score_, clf.best_estimator_)
    
    # Select the best model
    best_model_name, (best_score, best_estimator) = max(best_models.items(), key=lambda item: item[1][0])
    print(f"Best Model: {best_model_name} with accuracy {best_score}")
    
    # Test the best model
    y_pred = best_estimator.predict(X_test)
    print('Accuracy of our best model on the test set: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    
    return best_estimator, scaler

def get_clean_data():
  data = pd.read_csv("data/data.csv")
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data


def main():
  data = get_clean_data()

  model, scaler = create_model(data)

  with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
  with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
  

if __name__ == '__main__':
  main()