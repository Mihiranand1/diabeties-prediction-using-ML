import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('C:\\Users\\MIHIR\\Desktop\\diabetic\\diabetes.csv')
#print(diabetes_dataset.head() )

#print(diabetes_dataset['Outcome'].value_counts())

#print(diabetes_dataset.groupby('Outcome').mean())
# this shows the diff between diabetic and non diabetic

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
#print(standardized_data)# we transform all the 'X'value between 0-1 , so that it can give better prediction

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
#print(X.shape, X_train.shape, X_test.shape)

#Training the Model
classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

#print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

#print('Accuracy score of the test data : ', test_data_accuracy)

print("Now the model is trained\n")
print("there are 8 requirements to predict the Diabeties")

Pregnancies = int(input("Pregnancies : "))
Glucose = int(input("Glucose : "))
BloodPressure = int(input("BloodPressure : "))
SkinThickness = int(input("SkinThickness : "))
Insulin = int(input("Insulin : "))
BMI = float(input("BMI : "))
DiabetesPedigreeFunction = float(input("DiabetesPedigreeFunction : "))
Age = int(input("Age : "))


# Making a Predictive System

input_data = (Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
#print(std_data)

prediction = classifier.predict(std_data)
#print("predicted vlue = ",prediction)

if (prediction[0] == 0): #it is a list
  print('The person is not diabetic')
else:
  print('The person is diabetic')