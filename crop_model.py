import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import joblib

def train_save_model():
    df = pd.read_csv("crop_data.csv")
    x = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Model Accuracy:", round(accuracy_score(y_test, y_pred)*100))
    joblib.dump(model, 'crop_model.pkl')

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    model = joblib.load('crop_model.pkl')
    input_data = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    return model.predict(input_data)[0]

# # ---------------------------------------- Evaluation ----------------------------------------


# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", cm)

# # Classification Report (Precision, Recall, F1-score for each class)
# cr = classification_report(y_test, y_pred)
# print("Classification Report:\n", cr)
