import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
data = pd.read_csv("fake_job_postings.csv")

# Fill missing values
data = data.fillna('')

# Combine useful text fields into one
data['text'] = data['title'] + " " + data['description'] + " " + data['company_profile']

# Features and target
X = data['text']
y = data['fraudulent']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline (Text -> TF-IDF -> RandomForest)
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('classifier', RandomForestClassifier())
])

# Train model
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")
