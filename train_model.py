# === STEP 1: IMPORT TOOLS ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle

print("🤖 TRAINING AI MODEL")
print("="*50)

# === STEP 2: LOAD DATA ===
df = pd.read_csv('adverse_events_cleaned.csv')
print(f"\n📊 Loaded {len(df)} patient records")

# === STEP 3: PREPARE DATA FOR AI ===
print("\n🔧 Converting text and categories to numbers...")

# 3a. Convert reaction text to numbers (AI doesn't understand words)
vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
text_numbers = vectorizer.fit_transform(df['reactions'].fillna(''))
text_df = pd.DataFrame(text_numbers.toarray())

# 3b. Create age groups (child, adult, elderly)
df['age_group'] = pd.cut(df['patient_age'],
                          bins=[0, 18, 40, 65, 120],
                          labels=['child', 'adult', 'middle', 'elderly'])

# 3c. Convert categories to numbers
sex_numbers = pd.get_dummies(df['patient_sex'], prefix='sex')
age_numbers = pd.get_dummies(df['age_group'], prefix='age')

# 3d. Combine everything into one big table
all_features = pd.concat([text_df, sex_numbers, age_numbers], axis=1)
all_features.columns = all_features.columns.astype(str)
target = df['serious']  # What we're trying to predict

print(f"✅ Created {all_features.shape[1]} features (columns) for AI to learn from")

# === STEP 4: SPLIT DATA (80% learn, 20% test) ===
print("\n✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    all_features, target, test_size=0.2, random_state=42
)
print(f"Training set: {len(X_train)} patients")
print(f"Test set: {len(X_test)} patients")

# === STEP 5: TRAIN THE AI ===
print("\n🎓 Training AI model (this takes 10-30 seconds)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # AI learns patterns here
print("✅ Training complete!")

# === STEP 6: TEST THE AI ===
print("\n📝 Testing AI on patients it has never seen...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n🎯 RESULTS:")
print(f"   Accuracy: {accuracy:.1%}")
print(f"   Got {int(accuracy * len(X_test))} out of {len(X_test)} predictions correct")

# === STEP 7: SAVE EVERYTHING ===
print("\n💾 Saving model and predictions...")

# Save trained model
with open('severity_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Create predictions for ALL patients (for Power BI)
df['predicted_severity'] = model.predict(all_features)
df['confidence'] = model.predict_proba(all_features)[:, 1] * 100  # % confidence
df.to_csv('adverse_events_with_predictions.csv', index=False)

print("✅ Saved model as 'severity_model.pkl'")
print("✅ Saved predictions as 'adverse_events_with_predictions.csv'")
print("\n🎉 ALL DONE! You now have a trained AI model!")