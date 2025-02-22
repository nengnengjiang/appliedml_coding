import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example dataset
data = {
    'prompt': [
        "I love sunny days",
        "You should suffer!",
        "This is a polite discussion",
        "I hate your face, go away!",
        "Let's talk about books",
        "Disgusting code, you should die!",
        "You're a genius, thanks!",
        "I'll find you and you will regret it",
    ],
    'label': [0, 1, 0, 1, 0, 1, 0, 1],
    'timestamp': [
        "2025-01-01 10:00:00",
        "2025-01-01 10:05:00",
        "2025-01-02 12:30:00",
        "2025-01-02 13:45:00",
        "2025-01-03 08:15:00",
        "2025-01-03 23:00:00",
        "2025-01-04 09:00:00",
        "2025-01-04 22:50:00"
    ],
    'region': ["US", "US", "EU", "US", "APAC", "US", "APAC", "EU"],
    'device': ["mobile", "desktop", "mobile", "mobile", "desktop", "desktop", "mobile", "mobile"],
    'user_tenure': [100, 5, 365, 2, 50, 10, 200, 7]
}

df = pd.DataFrame(data)

# -------- Basic inspection

# 2.1 Look at the first few rows
print("DataFrame head:\n", df.head(), "\n")

# 2.2 Check column data types and non-null counts
print("DataFrame info:")
print(df.info(), "\n")

# 2.3 Check for null / missing values
print("Missing values per column:")
print(df.isnull().sum(), "\n")

# 2.4 Check for duplicates
num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}\n")

# 2.5 Convert timestamp to datetime (if not already)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2.6 Describe numeric columns
print("Numeric column stats:\n", df.describe(), "\n")

# -------- Label Distribution & Quick Categorical Checks

# 3.1 Check label distribution
print("Label distribution (counts):")
print(df['label'].value_counts(), "\n")

# Or normalized distribution
print("Label distribution (%):")
print(df['label'].value_counts(normalize=True), "\n")

# 3.2 Check region distribution
print("Region value counts:")
print(df['region'].value_counts(), "\n")

# 3.3 Check device distribution
print("Device value counts:")
print(df['device'].value_counts(), "\n")

# ---------Simple Explorations
# --------- Time-Related Patterns

# Extract hour/day from timestamp
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday

# Group by hour and see the average "label" (just as an example)
hourly_label = df.groupby('hour')['label'].mean()
print("Mean label by hour:\n", hourly_label, "\n")

# You could also do a count plot of "hour"
plt.figure(figsize=(6, 4))
sns.countplot(x='hour', data=df)
plt.title("Count of Posts by Hour of Day")
plt.show()

# ---------S Basic stats Numeric Feature
print("User tenure stats:\n", df['user_tenure'].describe(), "\n")
tenure_by_label = df.groupby('label')['user_tenure'].mean()
# Quick histogram
# plt.figure(figsize=(6,4))
# sns.histplot(df['user_tenure'], bins=5, kde=True)
# plt.title("Distribution of User Tenure (days)")
# plt.show()

# put numerica into bins and look at labels distribution by bins
# Example: Create bins for user_tenure
bins = [0, 10, 50, 100, df['user_tenure'].max() + 1]
labels = ["0-10", "10-50", "50-100", "100+"]
df['tenure_bin'] = pd.cut(df['user_tenure'], bins=bins, labels=labels, right=False)

print("\nTenure bins with label distribution:\n")
counts = df.groupby(['tenure_bin', 'label']).size().unstack(fill_value=0)
# Calculate ratio of label=1 within each bin
counts['ratio'] = counts[1] / (counts[0] + counts[1])

# ---------6 Basic exploration for text feature 
df['text_length'] = df['prompt'].apply(lambda x: len(x.split()))
print("Text length stats:")
print(df['text_length'].describe(), "\n")

# 1. Plot distribution of text lengths
# plt.figure(figsize=(6,4))
# sns.histplot(df['text_length'], bins=5, kde=True)
# plt.title("Distribution of Prompt Word Counts")
# plt.show()

# 2. Create bins for text length
max_len = df['text_length'].max()
bins = [0, 5, 10, 15, max_len + 1]
labels = ["0-5", "5-10", "10-15", f"15+ (up to {int(max_len)})"]
df['length_bin'] = pd.cut(df['text_length'], bins=bins, labels=labels, right=False)

# 3. Compute ratio of label=1 (harmful) within each text-length bin
counts_len = df.groupby(['length_bin', 'label']).size().unstack(fill_value=0)
counts_len['ratio'] = counts_len[1] / (counts_len[0] + counts_len[1])

print("Ratio of harmful content by text-length bin:\n")
print(counts_len['ratio'])

# 3. show what words are more frequently used in harmful content

from collections import Counter
import re
# Let's see frequent tokens overall
all_tokens = []
for row in df['prompt']:
    tokens = re.findall(r'\w+', row.lower())  # naive tokenizing
    all_tokens.extend(tokens)

common_words_all = Counter(all_tokens).most_common(10)
print("Top 10 most frequent tokens (overall):\n", common_words_all)

# Now focus on harmful content only (label=1)
harmful_df = df[df['label'] == 1]
harmful_tokens = []
for row in harmful_df['prompt']:
    tokens = re.findall(r'\w+', row.lower())
    harmful_tokens.extend(tokens)

common_words_harmful = Counter(harmful_tokens).most_common(10)
print("\nTop 10 most frequent tokens in harmful content (label=1):\n", common_words_harmful)




