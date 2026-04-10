import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("stu_dataset.csv")

# Convert Pass/Fail to numbers
data['Result'] = data['Result'].map({'Fail': 0, 'Pass': 1})

# Features & Target
X = data[['Study Hours', 'Attendance (%)', 'Assignments']]
y = data['Result']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Try your own input
study = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance %: "))
assignments = int(input("Enter Assignments completed: "))

prediction = model.predict([[study, attendance, assignments]])

if prediction == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")