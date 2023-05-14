# Importing the libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify

# the data is being loaded
df = pd.read_csv('winemag-data-130k-v2.csv')

# droping irrelevant columns
df = df.drop(['user_name', 'review_title', 'designation', 'region_2'], axis=1)

# droping rows with missing values as they are of no use , these are general processes which helps to work with all variety of dataset
df = df.dropna()

# Converting the categorical variables into numeric format
df = pd.get_dummies(df, columns=['country', 'province', 'region_1', 'winery', 'variety'])

# Spliting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('variety', axis=1), df['variety'], test_size=0.2, random_state=42)

# Training a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluating the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Building the Flask API,__name__ is the name of the current Python module
app = Flask(__name__)

# creates a simple route so we can see the application working
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = rf.predict(pd.DataFrame(data))
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
