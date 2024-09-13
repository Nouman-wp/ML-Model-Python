import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('https://raw.githubusercontent.com/Nouman-wp/ML-Model-/main/10%20%20pass%20Career%20Options%20in%20CSV%20%20.csv')
data = data.drop(columns=['Email Address'])
data = data.drop(columns=['Score'])
data = data.dropna()  # Handle missing values

# Calculating total score
data['Score'] = data.iloc[:, :-1].sum(axis=1)

#stream suggestion to studt based on the total score
def suggest_stream(score):
    if score >= 80:
        return 'Science'
    elif score >= 60:
        return 'Commerce'
    elif score >= 40:
        return 'Humanities'
    else:
        return 'Vocational Courses'

# Applying func
data['Suggested_Stream'] = data['Score'].apply(suggest_stream)

#distribution of streams
print("Unique classes in Suggested_Stream:")
print(data['Suggested_Stream'].value_counts())

# Features and target
X = data[['Score']]
y = data['Suggested_Stream']

#using encoder on target labels for the datatype
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Checking the unique values in y after encoding
print("Unique classes in y (encoded):")
print(pd.Series(y).value_counts())

# Spliting the dataset for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Checking the unique classes in y_train
print("Unique classes in y_train:")
print(pd.Series(y_train).value_counts())

# Train the model
if len(pd.Series(y_train).unique()) > 1:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)

    # Generate a random score for a new student
    new_student_score = np.random.randint(20, 101)  
    print(f"New student total score: {new_student_score}")

    # Predict the stream based on the new student's score
    predicted_stream = model.predict([[new_student_score]])
    predicted_stream_label = label_encoder.inverse_transform(predicted_stream)

    print(f"Suggested Stream for the new student: {predicted_stream_label[0]}")

    # Check if the prediction matches the rules
    def expected_stream(score):
        if score >= 80:
            return 'Science'
        elif score >= 60:
            return 'Commerce'
        elif score >= 40:
            return 'Humanities'
        else:
            return 'Vocational Courses'

    expected_stream_result = expected_stream(new_student_score)
    print(f"Expected Stream based on rules: {expected_stream_result}")

    # Check if the model's prediction aligns with the rules
    if predicted_stream_label[0] == expected_stream_result:
        print("The model's suggestion matches the expected stream.")
    else:
        print("The model's suggestion does NOT match the expected stream.")
else:
    print("Not enough distinct classes to train the model. Check the score distribution and stream assignment.")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
