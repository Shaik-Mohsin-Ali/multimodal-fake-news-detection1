import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model.text_model import predict_text

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

fake["label"] = "Fake"
real["label"] = "Real"

# Combine dataset
data = pd.concat([fake, real])

# Take small sample for faster testing
data = data.sample(200, random_state=42)

y_true = []
y_pred = []

print("Running evaluation...\n")

for index, row in data.iterrows():

    text = str(row["text"])[:512]
    true_label = row["label"]

    pred_label, score = predict_text(text)

    y_true.append(true_label)
    y_pred.append(pred_label)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label="Fake")
recall = recall_score(y_true, y_pred, pos_label="Fake")
f1 = f1_score(y_true, y_pred, pos_label="Fake")

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake","Real"],
            yticklabels=["Fake","Real"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

print("Model Evaluation Results")
print("------------------------")

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")