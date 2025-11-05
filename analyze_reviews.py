import pandas as pd
from transformers import pipeline
from pathlib import Path
from collections import Counter
import re

FILE_PATH = "f46a7ece-a8d2-4c71-a5ef-1cc9c3df8e3e.xlsx" 
df = pd.read_excel(FILE_PATH, sheet_name='in')
print("-CSV file reading complete")

# Keep necessary columns only
df = df[['product', 'categories', 'rating', 'reviews']].dropna(subset=['reviews'])

sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print("-Model loaded")

batch_size = 64
results = []
print("-Sentiment analysis started")
for i in range(0, len(df), batch_size):
    batch = df['reviews'].iloc[i:i+batch_size].astype(str).tolist()
    preds = sentiment_model(batch, truncation=True, max_length=512)
    results.extend(preds)

df['ml_sentiment'] = [r['label'] for r in results]
df['ml_confidence'] = [r['score'] for r in results]
print("-Sentiment analysis complete")


def split_cats(x):
    if pd.isna(x):
        return []
    return re.split(r"[|,;/]+", str(x).lower())

df['categories_list'] = df['categories'].apply(split_cats)
df_exp = df.explode('categories_list')

print("-Identifying best sellers across categories")
grouped = df_exp.groupby(['categories_list', 'product']).agg(
    review_count=('reviews', 'count'),
    avg_rating=('rating', 'mean'),
    pos_ratio=('ml_sentiment', lambda s: (s=='POSITIVE').sum() / len(s))
).reset_index()

bestsellers = grouped.sort_values(['categories_list', 'review_count'], ascending=[True, False]).groupby('categories_list').head(5)
bestsellers.to_csv("bestselling_by_category_ml.csv", index=False)

print("-Least selling products and negative feedback analysis")
prod_counts = df.groupby('product').size().reset_index(name='review_count').dropna(subset=['product'])
threshold = prod_counts['review_count'].quantile(0.10)
least_prods = prod_counts[prod_counts['review_count'] <= threshold]['product'].tolist()

least_analysis = []
for p in least_prods:
    subset = df[df['product']==p]
    negs = subset[subset['ml_sentiment']=='NEGATIVE']
    neg_ratio = len(negs) / len(subset)
    
    # Extract most common complaint words
    words = []
    for t in negs['reviews'].astype(str):
        words += re.findall(r"[a-zA-Z']+", t.lower())
    top_complaints = [w for w, _ in Counter(words).most_common(10)]
    
    actions = []
    if neg_ratio > 0.4:
        actions.append("Investigate product quality or description accuracy.")
    if any(w in top_complaints for w in ['price', 'expensive', 'costly']):
        actions.append("Re-evaluate pricing or offer discounts.")
    if any(w in top_complaints for w in ['broken', 'defective', 'damaged']):
        actions.append("Inspect supply chain & packaging process.")
    if not actions:
        actions.append("Gather more reviews for insights; monitor future feedback.")
    
    least_analysis.append({
        "product": p,
        "total_reviews": len(subset),
        "negative_ratio": round(neg_ratio, 3),
        "top_complaints": ", ".join(top_complaints[:10]),
        "suggested_actions": "; ".join(actions)
    })

least_df = pd.DataFrame(least_analysis)
least_df.to_csv("least_selling_analysis_ml.csv", index=False)

df.to_csv("sentiment_results_ml.csv", index=False)

print("-Files generated:")
print("-sentiment_results_ml.csv")
print("-bestselling_by_category_ml.csv")
print("-least_selling_analysis_ml.csv")

