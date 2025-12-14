import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load the dataset
df = pd.read_csv('amazon_alexa.csv')

print("="*80)
print("AMAZON ALEXA REVIEWS - EXPLORATORY DATA ANALYSIS")
print("="*80)

# ============================================================================
# 1. BASIC DATASET INFORMATION
# ============================================================================
print("\n1. BASIC DATASET INFORMATION")
print("-" * 80)
print(f"Dataset Shape: {df.shape}")
print(f"Total Records: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")
print("\nFirst 5 rows:")
print(df.head())

# ============================================================================
# 2. DATA TYPES AND STRUCTURE
# ============================================================================
print("\n2. DATA TYPES AND STRUCTURE")
print("-" * 80)
print(df.info())
print("\nColumn Data Types:")
print(df.dtypes)

# ============================================================================
# 3. SUMMARY STATISTICS
# ============================================================================
print("\n3. SUMMARY STATISTICS")
print("-" * 80)
print(df.describe())
print("\nAdditional Statistics:")
print(f"Rating Mean: {df['rating'].mean():.2f}")
print(f"Rating Median: {df['rating'].median():.2f}")
print(f"Rating Mode: {df['rating'].mode()[0]}")
print(f"Rating Std Dev: {df['rating'].std():.2f}")

# ============================================================================
# 4. MISSING VALUES ANALYSIS
# ============================================================================
print("\n4. MISSING VALUES ANALYSIS")
print("-" * 80)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df)

# ============================================================================
# 5. UNIQUE VALUES COUNT
# ============================================================================
print("\n5. UNIQUE VALUES COUNT")
print("-" * 80)
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# ============================================================================
# 6. TARGET VARIABLE DISTRIBUTION (FEEDBACK)
# ============================================================================
print("\n6. TARGET VARIABLE DISTRIBUTION (FEEDBACK)")
print("-" * 80)
print(df['feedback'].value_counts())
print("\nFeedback Distribution (%):")
print(df['feedback'].value_counts(normalize=True) * 100)

# ============================================================================
# 7. RATING DISTRIBUTION
# ============================================================================
print("\n7. RATING DISTRIBUTION")
print("-" * 80)
print(df['rating'].value_counts().sort_index())

# ============================================================================
# 8. VARIATION ANALYSIS
# ============================================================================
print("\n8. VARIATION (PRODUCT TYPE) ANALYSIS")
print("-" * 80)
print(df['variation'].value_counts())

# ============================================================================
# 9. CORRELATION ANALYSIS
# ============================================================================
print("\n9. CORRELATION ANALYSIS")
print("-" * 80)
correlation = df[['rating', 'feedback']].corr()
print(correlation)

# ============================================================================
# 10. GROUPED AGGREGATIONS - RATING BY FEEDBACK
# ============================================================================
print("\n10. RATING STATISTICS BY FEEDBACK")
print("-" * 80)
rating_by_feedback = df.groupby('feedback')['rating'].agg(['mean', 'median', 'std', 'count'])
print(rating_by_feedback)

# ============================================================================
# 11. GROUPED AGGREGATIONS - FEEDBACK BY VARIATION
# ============================================================================
print("\n11. FEEDBACK DISTRIBUTION BY PRODUCT VARIATION")
print("-" * 80)
feedback_by_variation = pd.crosstab(df['variation'], df['feedback'], normalize='index') * 100
print(feedback_by_variation)

# ============================================================================
# 12. REVIEW LENGTH ANALYSIS
# ============================================================================
print("\n12. REVIEW LENGTH ANALYSIS")
print("-" * 80)
df['review_length'] = df['verified_reviews'].astype(str).apply(len)
df['word_count'] = df['verified_reviews'].astype(str).apply(lambda x: len(x.split()))
print(f"Average Review Length (characters): {df['review_length'].mean():.2f}")
print(f"Average Word Count: {df['word_count'].mean():.2f}")
print(f"Max Review Length: {df['review_length'].max()}")
print(f"Min Review Length: {df['review_length'].min()}")

review_stats = df.groupby('feedback')[['review_length', 'word_count']].mean()
print("\nReview Statistics by Feedback:")
print(review_stats)

# ============================================================================
# 13. OUTLIER DETECTION
# ============================================================================
print("\n13. OUTLIER DETECTION")
print("-" * 80)
Q1 = df['rating'].quantile(0.25)
Q3 = df['rating'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['rating'] < Q1 - 1.5 * IQR) | (df['rating'] > Q3 + 1.5 * IQR)]
print(f"Number of outliers in Rating: {len(outliers)}")
print(f"Percentage of outliers: {(len(outliers)/len(df))*100:.2f}%")

# ============================================================================
# 14. TIME SERIES ANALYSIS (DATE)
# ============================================================================
print("\n14. TIME SERIES ANALYSIS")
print("-" * 80)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

print("Reviews by Year:")
print(df['year'].value_counts().sort_index())

# ============================================================================
# 15. FEEDBACK DISTRIBUTION BY RATING
# ============================================================================
print("\n15. FEEDBACK DISTRIBUTION BY RATING")
print("-" * 80)
feedback_rating = pd.crosstab(df['rating'], df['feedback'])
print(feedback_rating)

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

# Handle missing values
df = df.dropna(subset=['verified_reviews'])
print(f"\nDataset shape after removing missing reviews: {df.shape}")

# Prepare features
X = df['verified_reviews'].astype(str)
y = df['feedback']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Text Vectorization using TF-IDF
print("\nPerforming TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
# ============================================================================
# MACHINE LEARNING MODEL
# ============================================================================
print("\n" + "="*80)
print("MACHINE LEARNING MODEL TRAINING")
print("="*80)

# Train Random Forest Classifier
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature Importance (Top 20 words)
feature_importance = pd.DataFrame({
    'feature': vectorizer.get_feature_names_out(),
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

print("\nTop 20 Most Important Features (Words):")
print(feature_importance)

# ============================================================================
# SAVE MODEL AND VECTORIZER
# ============================================================================
import joblib

print("\n" + "="*80)
print("SAVING MODEL AND VECTORIZER")
print("="*80)

joblib.dump(model, 'alexa_sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
df.to_csv('processed_alexa_data.csv', index=False)

print("\nModel saved as: alexa_sentiment_model.pkl")
print("Vectorizer saved as: tfidf_vectorizer.pkl")
print("Processed data saved as: processed_alexa_data.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)