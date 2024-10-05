import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('https://raw.githubusercontent.com/Nouman-wp/ML-Model-/main/10%20%20pass%20Career%20Options%20in%20CSV%20%20.csv')
data = data.drop(columns=['Email Address'])
data = data.drop(columns=['Score'])
data = data.dropna()  


data['Score'] = data.iloc[:, :-1].sum(axis=1)


def suggest_stream(score):
    if score >= 80:
        return 'Science'
    elif score >= 60:
        return 'Commerce'
    elif score >= 40:
        return 'Humanities'
    else:
        return 'Vocational Courses'


data['Suggested_Stream'] = data['Score'].apply(suggest_stream)


print("Unique classes in Suggested_Stream:")
print(data['Suggested_Stream'].value_counts())


X = data[['Score']]
y = data['Suggested_Stream']


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


print("Unique classes in y (encoded):")
print(pd.Series(y).value_counts())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Unique classes in y_train:")
print(pd.Series(y_train).value_counts())


if len(pd.Series(y_train).unique()) > 1:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    new_student_score = np.random.randint(20, 101)  
    print(f"New student total score: {new_student_score}")


    predicted_stream = model.predict([[new_student_score]])
    predicted_stream_label = label_encoder.inverse_transform(predicted_stream)

    print(f"Suggested Stream for the new student: {predicted_stream_label[0]}")


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


    if predicted_stream_label[0] == expected_stream_result:
        print("The model's suggestion matches the expected stream.")
    else:
        print("The model's suggestion does NOT match the expected stream.")
else:
    print("Not enough distinct classes to train the model. Check the score distribution and stream assignment.")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
