import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
df = pd.read_csv('data/telco_churn.csv')
#dropping coustomerId as it is not useful in prediction as it is just an identifier
df.drop('customerID', axis=1, inplace=True)
#using lableEncoder endcoding categorical columns into numbers.
for column in df.select_dtypes(include=['object']).columns:
    df[column] = LabelEncoder().fit_transform(df[column])
#deviding data into x(all features:independent features) and y(churn:1 for churned and 0 for not churned)
x= df.drop('Churn', axis=1)
y= df['Churn']
#splitting the data set into training(80%) and testing(20%) sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)
#saving the model as .pkl file for streamlit app
joblib.dump(model,'churn_model.pkl')
#evaluating the model
y_pred =model.predict(x_test)
print(classification_report(y_test, y_pred))