import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('dataset/student_data.csv')

X = data[['study_hours', 'attendance', 'assignments_completed']]
y = data['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)