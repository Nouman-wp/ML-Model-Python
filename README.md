# Student Stream Suggestion Model

This project predicts a suitable stream (Science, Commerce, Humanities, or Vocational Courses) for students based on their scores. The model uses logistic regression to classify students based on their total score, and also validates its prediction using a simple rule-based approach. The dataset used includes students' scores in various subjects, and their stream is suggested based on a pre-defined scoring threshold.

## Dataset
The dataset used for this project can be found [here](https://raw.githubusercontent.com/Nouman-wp/ML-Model-/main/10%20%20pass%20Career%20Options%20in%20CSV%20%20.csv). It includes the following:
- Subject scores for each student.
- **Note:** The dataset excludes the "Email Address" column to maintain privacy and focuses only on subject scores.

## How the Model Works
1. **Score Calculation**: 
   - The total score is computed by summing individual subject scores for each student.
   
2. **Stream Suggestion**: 
   - Based on the total score, a stream is suggested:
     - Science: Total score ≥ 80
     - Commerce: 60 ≤ Total score < 80
     - Humanities: 40 ≤ Total score < 60
     - Vocational Courses: Total score < 40

3. **Logistic Regression**:
   - The data is split into training and testing sets.
   - Logistic regression is used to train a model for predicting the stream based on the total score.

4. **Model Evaluation**:
   - The model is evaluated using the accuracy score and compares the model’s predictions against the rule-based expected streams.

5. **New Student Prediction**:
   - A new random student score is generated, and the model predicts the appropriate stream.
   - The prediction is cross-checked with the rule-based stream assignment to ensure consistency.

## Code Summary

### 1. **Data Loading and Preprocessing**:
   - The dataset is loaded and unnecessary columns, such as email addresses, are dropped.
   - Missing values are handled, and a total score is calculated for each student.

### 2. **Stream Assignment**:
   - A function `suggest_stream` is used to assign streams based on the total score.

### 3. **Label Encoding**:
   - The target labels (streams) are encoded using `LabelEncoder` to prepare for training.

### 4. **Model Training**:
   - The dataset is split into training and testing sets, and a logistic regression model is trained.

### 5. **Prediction and Validation**:
   - A random score for a new student is generated, and the stream prediction is validated against the rule-based assignment.

### 6. **Evaluation**:
   - The model’s performance is evaluated with accuracy and checked for consistency between the predicted and expected streams.

## Installation and Usage
1. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn
   ```
2. Run the script:
   ```bash
   python stream_suggestion.py
   ```
3. The script will output:
   - Stream distribution in the dataset.
   - Model accuracy.
   - Suggested streams for new students.
   - Comparison between model prediction and rule-based suggestions.

## Conclusion
This model effectively predicts a student's stream based on their score and validates the predictions with rule-based logic. The logistic regression model helps automate the decision-making process, while cross-checking ensures reliability.
