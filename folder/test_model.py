import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
akg_bayi_df = pd.read_csv(r'C:\Users\user\Downloads\TestModel\AKG-bayi.csv')
food_check_df = pd.read_csv(r'C:\Users\user\Downloads\TestModel\food-check.csv')


# Input testing data (Contoh input)
age = 2
foods = "Milk"
descriptions = "Milk, whole"

# Filter data based on input
akg_bayi_filtered = akg_bayi_df[akg_bayi_df['age'] == age]
food_check_filtered = food_check_df[
    (food_check_df['Category'] == foods) &
    (food_check_df['Description'] == descriptions)
]

if akg_bayi_filtered.empty or food_check_filtered.empty:
    raise ValueError("Data tidak ditemukan untuk input yang diberikan.")

# Align column names for common nutrition metrics
common_columns = [col for col in akg_bayi_filtered.columns if col in food_check_filtered.columns and col != 'age']
if not common_columns:
    raise ValueError("Tidak ada kolom nutrisi yang sesuai antara data AKG dan makanan.")

akg_bayi_filtered = akg_bayi_filtered[common_columns]
food_check_filtered = food_check_filtered[common_columns]

# Calculate nutrient differences
required_nutrients = akg_bayi_filtered.iloc[0].round(2)
consumed_nutrients = food_check_filtered.iloc[0].round(2)
nutrient_differences = (required_nutrients - consumed_nutrients).round(2)

# Normalize data
scaler = MinMaxScaler()
normalized_food_data = pd.DataFrame(scaler.fit_transform(food_check_df[common_columns]),
                                    columns=common_columns)
normalized_nutrient_diff = scaler.transform(nutrient_differences.values.reshape(1, -1))

# Get predictions for testing
similarities = cosine_similarity(normalized_nutrient_diff, normalized_food_data)
food_check_df['Relevance'] = similarities.flatten()

# Recommend top N foods
top_n = 5
recommended_foods = food_check_df.sort_values(by='Relevance', ascending=False).head(top_n)

# Display testing results
print("\nNutrisi yang harus dipenuhi:")
print(required_nutrients)

print("\nNutrisi makanan yang telah dikonsumsi:")
print(consumed_nutrients)

print("\nSelisih nutrisi:")
print(nutrient_differences)

print("\nRekomendasi makanan berdasarkan Content-Based Filtering:")
print(recommended_foods[['Category', 'Description', 'Relevance']])

