import pandas as pd
import json

# Load raw data
with open('fda_raw_data.json', 'r') as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} records")

# Extract key fields
records = []

for report in raw_data:
    try:
        # Basic info
        record = {
            'report_id': report.get('safetyreportid', 'Unknown'),
            'receive_date': report.get('receivedate', 'Unknown'),
            'serious': report.get('serious', 0),  # 1=serious, 2=non-serious
        }

        # Patient info
        patient = report.get('patient', {})
        record['patient_age'] = patient.get('patientonsetage', None)
        record['patient_sex'] = patient.get('patientsex', 'Unknown')

        # Drug info (take first drug listed)
        drugs = patient.get('drug', [])
        if drugs:
            record['drug_name'] = drugs[0].get('medicinalproduct', 'Unknown')
        else:
            record['drug_name'] = 'Unknown'

        # Reactions (combine all reactions into one text)
        reactions = patient.get('reaction', [])
        reaction_list = [r.get('reactionmeddrapt', '') for r in reactions]
        record['reactions'] = ', '.join(reaction_list)

        # Outcome
        record['outcome'] = report.get('seriousnessdeath', 0)  # Example outcome

        records.append(record)

    except Exception as e:
        continue

# Create DataFrame
df = pd.DataFrame(records)

print("\n📊 Data Summary:")
print(df.info())
print("\n" + "=" * 50)
print(df.head())

# Basic cleaning
df = df[df['patient_age'].notna()]  # Remove records without age
df = df[df['reactions'] != '']  # Remove records without reactions
df['receive_date'] = pd.to_datetime(df['receive_date'], format='%Y%m%d', errors='coerce')
df = df[df['receive_date'].dt.year >= 2014]
df = df.dropna(subset=['receive_date'])
# convert to be numbers (handles strings, floats, and NaNs)
df['patient_age'] = pd.to_numeric(df['patient_age'], errors='coerce')
df = df[(df['patient_age'] > 0) & (df['patient_age'] < 120)]  # Reasonable age range
#Convert to numeric first (handles strings like "1" or "2")
temp_outcome = pd.to_numeric(df['outcome'], errors='coerce')
# STRICT CHECK: Only a literal 1 counts as Fatal.
# Everything else (0, 2, NaN, "Unknown") becomes 0.
df['outcome'] = (temp_outcome == 1).astype(int)
df['serious'] = pd.to_numeric(df['serious'], errors='coerce').fillna(0).astype(int)

# 2. Map them to simple numbers: 1 for Male, 2 for Female, 0 for Unknown
# This is called 'Encoding'
gender_map = {'1': 1, '2': 2, 'Male': 1, 'Female': 2}
df['patient_sex'] = df['patient_sex'].map(gender_map).fillna(0).astype(int)

# Convert serious to binary (1=serious, 0=non-serious)
df['serious'] = df['serious'].apply(lambda x: 1 if x == 1 else 0)

print(f"\n✅ Cleaned data: {len(df)} records")
print(f"Serious cases: {df['serious'].sum()} ({df['serious'].mean() * 100:.1f}%)")
print(f"Non-serious cases: {(1 - df['serious']).sum()} ({(1 - df['serious'].mean()) * 100:.1f}%)")
print(f"Total fatalities in this batch: {df['outcome'].sum()}")

#The "Quick Look" Correlation
correlation = df['patient_sex'] .corr(df['serious'])
print(f"Correlation between patient sex and Seriousness: {correlation:.4f}")

# Save cleaned data
df.to_csv('adverse_events_cleaned.csv', index=False)

print("\n✅ Saved to adverse_events_cleaned.csv")
