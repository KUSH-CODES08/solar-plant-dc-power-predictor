# ============================================================
# IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from datetime import datetime
import joblib


# ============================================================
# STEP 1: LOADING DATASETS
# ============================================================
print("=" * 80)
print("STEP 1: LOADING DATASETS")
print("=" * 80)

generation_data = pd.read_csv(
    'week-2\\SOLAR POWER GENERATION\\Plant_1_Generation_Data.csv'
)
weather_data = pd.read_csv(
    'week-2\\SOLAR POWER GENERATION\\Plant_1_Weather_Sensor_Data.csv'
)

print("\n✅ Generation Data Loaded")
print("Rows:", generation_data.shape[0])
print("Columns:", generation_data.shape[1])

print("\n✅ Weather Data Loaded")
print("Rows:", weather_data.shape[0])
print("Columns:", weather_data.shape[1])


# ============================================================
# STEP 2: INITIAL DATA OVERVIEW
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: DATA OVERVIEW")
print("=" * 80)

print("\n📌 Generation Data Sample")
print(generation_data.head())

print("\n📌 Weather Data Sample")
print(weather_data.head())


# ============================================================
# STEP 3: DATE_TIME CONVERSION
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: DATE_TIME CONVERSION")
print("=" * 80)

generation_data['DATE_TIME'] = pd.to_datetime(
    generation_data['DATE_TIME'], dayfirst=True
)
weather_data['DATE_TIME'] = pd.to_datetime(
    weather_data['DATE_TIME'], dayfirst=True
)

print("✅ DATE_TIME converted successfully")


# ============================================================
# STEP 4: AGGREGATING GENERATION DATA
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: AGGREGATING GENERATION DATA")
print("=" * 80)

generation_agg = generation_data.groupby('DATE_TIME').agg({
    'DC_POWER': 'sum',
    'AC_POWER': 'sum',
    'DAILY_YIELD': 'mean',
    'TOTAL_YIELD': 'mean'
}).reset_index()

print("✅ Generation data aggregated")
print(generation_agg.head())


# ============================================================
# STEP 5: AGGREGATING WEATHER DATA
# ============================================================
print("\n" + "=" * 80)
print("STEP 5: AGGREGATING WEATHER DATA")
print("=" * 80)

weather_agg = weather_data.groupby('DATE_TIME').agg({
    'AMBIENT_TEMPERATURE': 'mean',
    'MODULE_TEMPERATURE': 'mean',
    'IRRADIATION': 'mean'
}).reset_index()

print("✅ Weather data aggregated")
print(weather_agg.head())


# ============================================================
# STEP 6: MERGING DATASETS
# ============================================================
print("\n" + "=" * 80)
print("STEP 6: MERGING DATASETS")
print("=" * 80)

Final_Data = pd.merge(
    generation_agg,
    weather_agg,
    on='DATE_TIME',
    how='inner'
)

print("✅ Final dataset created")
print("Rows:", Final_Data.shape[0])
print("Columns:", Final_Data.shape[1])


# ============================================================
# STEP 7: SAVING FINAL DATASET
# ============================================================
Final_Data.to_csv('Plant1_Merged_Dataset.csv', index=False)
print("\n💾 Final dataset saved as Plant1_Merged_Dataset.csv")


# ============================================================
# DATA ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("DATA ANALYSIS STARTED")
print("=" * 80)

print("\n🔍 NULL VALUES CHECK")
print(Final_Data.isnull().sum())


# Histogram
print("\n📊 IRRADIATION DISTRIBUTION")
plt.hist(Final_Data["IRRADIATION"], bins=50)
plt.title("IRRADIATION Distribution")
plt.show()


# Daytime filtering
daytime_data = Final_Data[Final_Data['IRRADIATION'] > 0]

print("\n☀️ DAYTIME DATA FILTERING")
print("Total Records:", Final_Data.shape[0])
print("Daytime Records:", daytime_data.shape[0])


# Feature Distributions
numeric_columns = daytime_data.select_dtypes(
    include=['float64', 'int64']
).columns

sns.set(style="whitegrid")
plt.figure(figsize=(15, 12))

print("\n📈 FEATURE DISTRIBUTIONS (DAYTIME DATA)")
for i, col in enumerate(numeric_columns):
    plt.subplot(3, 3, i + 1)
    sns.histplot(daytime_data[[col]], kde=True, bins=50)
    plt.title(col)

plt.tight_layout()
plt.show()


# ============================================================
# OUTLIER DETECTION & REMOVAL
# ============================================================
Features = [
    'IRRADIATION',
    'MODULE_TEMPERATURE',
    'AMBIENT_TEMPERATURE',
    'DAILY_YIELD',
    'DC_POWER'
]

print("\n🚨 OUTLIER DETECTION")
sns.boxplot(data=Final_Data[Features])
plt.title("Before Outlier Removal")
plt.show()

Data_Clean = Final_Data.copy()

for col in Features:
    Q1 = Data_Clean[col].quantile(0.25)
    Q3 = Data_Clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    Data_Clean = Data_Clean[
        (Data_Clean[col] >= lower) & (Data_Clean[col] <= upper)
    ]

print("\n🧹 OUTLIERS REMOVED")
print("Rows before:", Final_Data.shape[0])
print("Rows after:", Data_Clean.shape[0])

sns.boxplot(data=Data_Clean[Features])
plt.title("After Outlier Removal")
plt.show()


# ============================================================
# TIME SERIES ANALYSIS
# ============================================================
Final_Data.set_index('DATE_TIME', inplace=True)
Final_Data.sort_index(inplace=True)

print("\n⏱️ TIME SERIES PLOT (DC_POWER)")
plt.figure(figsize=(12, 6))
plt.plot(Final_Data.index, Final_Data['DC_POWER'], label="DC_POWER")
plt.legend()
plt.show()


# ============================================================
# CORRELATION ANALYSIS
# ============================================================
print("\n🔗 CORRELATION MATRIX")
correlation_matrix = Final_Data.corr(numeric_only=True)
print(correlation_matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="viridis", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ================================
# FEATURE SCALING (INPUT FEATURES ONLY)
# ================================
input_features = [
    'IRRADIATION',
    'MODULE_TEMPERATURE',
    'AMBIENT_TEMPERATURE'
]

scaler = StandardScaler()
Final_Data[input_features] = scaler.fit_transform(
    Final_Data[input_features]
)

# ============================================================
# MODEL TRAINING
# ============================================================
features = [
    'IRRADIATION',
    'MODULE_TEMPERATURE',
    'AMBIENT_TEMPERATURE'
]

X = Final_Data[input_features]
y = Final_Data['DC_POWER']   # ⚠️ NOT scaled


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=40
)

print("\n🧪 TRAIN-TEST SPLIT")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

model = LinearRegression()
model.fit(X,y)

print("\n🤖 MODEL TRAINED (Linear Regression)")


# ============================================================
# MODEL EVALUATION
# ============================================================  
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 MODEL PERFORMANCE")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)


# ============================================================
# MODEL SAVING
# ============================================================
joblib.dump(model, 'solar_Power_generation_model.pkl')
joblib.dump(scaler, "scaler.pkl")

print("\n💾 MODEL SAVED SUCCESSFULLY")
print("File: solar_Power_generation_model.pkl")
print("\n💾 SCALER SAVED SUCCESSFULLY")
print("File: scaler.pkl")
print("\n🚀 PROJECT COMPLETED SUCCESSFULLY")
print("=" * 80)


