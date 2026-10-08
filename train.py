import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Dataset download aur clean ho raha hai...")
url = "https://raw.githubusercontent.com/dataprofessor/data/master/heart-disease-cleveland.csv"
df = pd.read_csv(url)

# Columns ke extra spaces remove karein
df.columns = df.columns.str.strip()

# Missing values ya '?' ko NaN mein convert karke drop/fill karein
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Features (X) aur Target (y) alag karein (Aakhri column 'diagnosis' target hai)
X = df.iloc[:, :-1]
y = df.iloc[:, -1].astype(int)  # Ensure target is integer

# Train aur Test data mein split karein
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Model train ho raha hai...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model ko save karein
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Success! Model train ho gaya aur 'model.pkl' successfully save ho gaya hai.")