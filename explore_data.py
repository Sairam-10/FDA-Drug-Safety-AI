import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('adverse_events_cleaned.csv')

print("📊 EXPLORATORY DATA ANALYSIS\n" + "="*50)

# 1. Age distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df['patient_age'], bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')

# 2. Serious vs Non-Serious
plt.subplot(1, 3, 2)
df['serious'].value_counts().plot(kind='bar')
plt.xlabel('Serious (1) vs Non-Serious (0)')
plt.ylabel('Count')
plt.title('Event Severity')
plt.xticks(rotation=0)

# 3. Top 10 Most Frequently Reported Drugs
plt.subplot(1, 3, 3)
top_drugs = df['drug_name'].value_counts().head(10)
top_drugs = top_drugs.sort_values(ascending=True)
top_drugs.plot(kind='barh')
plt.xlabel('Count')
plt.title('Top 10 Most Frequently Reported Drugs')

plt.tight_layout()
plt.savefig('eda_charts.png', dpi=150, bbox_inches='tight')
print("✅ Charts saved to eda_charts.png")

# Summary stats
print("\n📈 Key Statistics:")
print(f"Average age: {df['patient_age'].mean():.1f} years")
print(f"Most frequently reported drug: {df['drug_name'].value_counts().index[0]}")
print(f"Most common reaction: {df['reactions'].value_counts().index[0]}")

plt.show()