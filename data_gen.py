import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker to generate fake data
fake = Faker()

# Generate 100 records
num_records = 1000
data = []

# Generate data for each record
for _ in range(num_records):
    state = fake.state()
    job_start_date = fake.date_between(start_date='-1y', end_date='today').strftime('%Y-%m-%d')
    scrape_sale_qty = random.randint(50, 150)  # Random quantity between 50 and 150
    rate_per_kg = random.uniform(40, 60)  # Random rate per kg between 40 and 60
    
    # Introduce outliers intentionally
    if random.random() < 0.1:  # 10% chance of introducing outliers
        if random.random() < 0.5:
            scrape_sale_qty = random.randint(200, 300)  # Introduce high outlier
        else:
            scrape_sale_qty = random.randint(10, 20)   # Introduce low outlier
    
    amount_in_rs = scrape_sale_qty * rate_per_kg
    
    # Format amount to 2 decimal places
    amount_in_rs = round(amount_in_rs, 2)
    
    # Append to data list
    data.append([state, job_start_date, scrape_sale_qty, rate_per_kg, amount_in_rs])

# Create a DataFrame
df = pd.DataFrame(data, columns=['State', 'Job Start Date', 'Scrape Sale Qty', 'Rate Rs/Kg', 'Amount in Rs'])

# Display first few records to verify
print(df.head())

# Save to CSV if needed
df.to_csv('data/dataset_with_outliers.csv', index=False)