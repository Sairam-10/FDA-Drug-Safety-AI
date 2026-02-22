import requests
import pandas as pd
import json

print("Fetching FDA adverse event data...")

# FDA API - Public, free, no API key needed
url = "https://api.fda.gov/drug/event.json"

# Fetch 5000 records (split into batches to avoid API limits)
all_data = []

for skip in range(0, 5000, 100):  # Fetch in batches of 100
    params = {
        'limit': 100,
        'skip': skip
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            all_data.extend(data['results'])
            print(f"Fetched {len(all_data)} records so far...")
        else:
            print(f"Error at skip={skip}")
    except Exception as e:
        print(f"Error: {e}")
        continue

print(f"\nTotal records fetched: {len(all_data)}")

# Save raw data
with open('fda_raw_data.json', 'w') as f:
    json.dump(all_data, f)

print("✅ Data saved to fda_raw_data.json")