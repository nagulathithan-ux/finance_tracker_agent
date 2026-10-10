import csv
import random
from datetime import date, timedelta

random.seed(7)

rows = []
start = date(2026, 8, 1)

def add(d, desc, amount):
    rows.append([d.isoformat(), desc, amount])

# Income
add(start, "SALARY CREDIT - EMPLOYER PVT LTD", 65000)
add(start + timedelta(days=30), "SALARY CREDIT - EMPLOYER PVT LTD", 65000)

# Rent
add(start + timedelta(days=2), "RENT PAYMENT TO LANDLORD", -15000)
add(start + timedelta(days=32), "RENT PAYMENT TO LANDLORD", -15000)

# Recurring subscriptions (same desc/amount monthly -> recurring)
for m in [0, 30]:
    add(start + timedelta(days=5 + m), "NETFLIX SUBSCRIPTION", -499)
    add(start + timedelta(days=6 + m), "SPOTIFY PREMIUM", -119)
    add(start + timedelta(days=7 + m), "AMAZON PRIME MEMBERSHIP", -299)

# Groceries (several, regular range)
grocery_desc = ["BIGBASKET GROCERY ORDER", "DMART SUPERMARKET", "RELIANCE FRESH PURCHASE"]
for i in range(10):
    d = start + timedelta(days=random.randint(0, 55))
    add(d, random.choice(grocery_desc), -round(random.uniform(400, 1800), 2))

# Food & Dining
food_desc = ["SWIGGY FOOD ORDER", "ZOMATO FOOD DELIVERY", "STARBUCKS COFFEE", "CAFE COFFEE DAY"]
for i in range(14):
    d = start + timedelta(days=random.randint(0, 55))
    add(d, random.choice(food_desc), -round(random.uniform(150, 900), 2))

# Transportation
trans_desc = ["UBER RIDE", "OLA CAB BOOKING", "PETROL PUMP FUEL", "METRO CARD RECHARGE"]
for i in range(10):
    d = start + timedelta(days=random.randint(0, 55))
    add(d, random.choice(trans_desc), -round(random.uniform(100, 1200), 2))

# Utilities
add(start + timedelta(days=10), "ELECTRICITY BILL PAYMENT", -2200)
add(start + timedelta(days=40), "ELECTRICITY BILL PAYMENT", -2450)
add(start + timedelta(days=12), "BROADBAND WIFI BILL", -999)
add(start + timedelta(days=42), "BROADBAND WIFI BILL", -999)
add(start + timedelta(days=15), "MOBILE RECHARGE", -449)

# Shopping
shop_desc = ["AMAZON.IN PURCHASE", "FLIPKART ORDER", "MYNTRA FASHION STORE"]
for i in range(6):
    d = start + timedelta(days=random.randint(0, 55))
    add(d, random.choice(shop_desc), -round(random.uniform(500, 2500), 2))

# Entertainment
add(start + timedelta(days=20), "PVR CINEMAS MOVIE TICKET", -450)
add(start + timedelta(days=48), "BOOKMYSHOW CONCERT TICKET", -1800)

# Healthcare
add(start + timedelta(days=25), "APOLLO PHARMACY", -650)

# Anomaly: one very large one-off purchase (spike)
add(start + timedelta(days=18), "AMAZON.IN PURCHASE - LAPTOP", -68000)

# A duplicate row on purpose, to test de-duplication
rows.append(rows[-1])

# A row with a missing/bad date, to test cleaning
rows.append(["not-a-date", "CORRUPTED ROW TEST", -100])

# A row with a missing amount, to test cleaning
rows.append([(start + timedelta(days=50)).isoformat(), "MISSING AMOUNT TEST", ""])

with open("/home/claude/finance_tracker_agent/data/sample_transactions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["date", "description", "amount"])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows (including 3 deliberately dirty rows for cleaning tests)")
