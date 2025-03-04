'''
Recruiter prompt:

For the technical portion, it will focus on evaluating the following: 
	• Standard ML process (train/test split, looking at data, model training, etc)
	• Model evaluation and threshold selection
	• A bit of model deployment (monitoring, what can go wrong)
	

https://www.1point3acres.com/bbs/thread-1073090-5-1.html

ML coding, 给一个数据集，然后要求训练一个模型; 数据里面有些坑，训练出来之后还要进行 debug，之后问了一些 model deployment 以及在 production 里面如何去使用的问题，比较 open end

就是给了一个 prompte 的数据集，然后做分类看哪些是 harmful 的哪些不是 harmful 的 经典的二分类问题

我就是用的传统的 ML 的方法解决的问题，所有的 code 都要自己写，它们给的只有个数据集，除此之外什么也没有

可以调用一些库，我当时使用的就是 sklearn 里面的

'''

'''
Example data created by chatGPT

data = {
    'prompt': [
        "I hate your face, you should suffer!",
        "I love sunny days!",
        "Te voy a lastimar si no haces caso!",
        "Thanks for your code, it was helpful."
    ],
    'label': [1, 0, 1, 0],
    'timestamp': [
        "2025-01-01 14:32:00",
        "2025-01-01 09:15:00",
        "2025-01-02 21:45:00",
        "2025-01-03 06:10:00"
    ],
    'server_name': ["us-east-1a", "eu-west-1a", "us-east-1a", "ap-south-1b"],
    'region': ["US", "EU", "US", "APAC"],
    'ip_address': ["192.168.1.10", "10.0.0.55", "8.8.8.8", "172.16.0.1"],
    'device': ["mobile", "desktop", "mobile", "mobile"],
    'user_tenure': [10, 365, 2, 100]
}


Common “Pitfalls” or “Gotchas” in the Data
Some harmful messages might sound polite at first but carry hateful undertones.
Multilingual data
Some not harmful messages might contain strong words but are not actually harmful.
There could be duplicates or near-duplicates.
Imbalanced data treatment and proper evaluation methods
Some malicious text might be hidden in a “nice-sounding” message, or it might contain acronyms, emojis, or special characters



Below is an end-to-end code that:

Loads the data .
Explores it briefly.
Splits into train/test sets.
Builds a pipeline for text preprocessing and a proper model should be used.
Trains, evaluates, and explores threshold selection.
Discusses potential deployment strategies and monitoring.

'''

import pandas as pd
import numpy as np
# import torch

# -----------------------------------------------------------
# 1. LOAD THE DATA
# -----------------------------------------------------------
data = pd.read_csv('harmful_data.csv')  # Replace with your actual path


# load other format json file

import json
with open("fake_dataset_10k.json", "r") as f:
    data = json.load(f)
    # data = [json.loads(line) for line in file]
print(len(data), "entries loaded.")
print(data[0])
# 2. Convert to DataFrame
df = pd.DataFrame(data)
# 3. Optional: Extract the text fields from 'prompt' and 'completion'
  #  Since 'prompt' is a list of dicts (e.g., [{"role": "user", "content": "..."}]),
  #  we'll grab the 'content' of the first item.
df["prompt_content"] = df["prompt"].apply(
    lambda lst: lst[0]["content"] if isinstance(lst, list) and len(lst) > 0 else None
)
df["completion_content"] = df["completion"].apply(
    lambda lst: lst[0]["content"] if isinstance(lst, list) and len(lst) > 0 else None
)
# 4. (Optional) Drop the original 'prompt' and 'completion' columns if you prefer
df.drop(columns=["prompt", "completion"], inplace=True)
# 5. Inspect the DataFrame
print(df.head())

#df = pd.read_json("data.json")
#df = pd.read_json("data.json", lines=True)

# -----------------------------------------------------------
# 2. QUICK EXPLORATION
# -----------------------------------------------------------
data.head()
df.info()
data.dtypes.value_counts()
data['label'].value_counts()

# Timestamp to Day/Hour
# Convert timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
# Example features
df['hour'] = df['timestamp'].dt.hour        # numeric
df['dayofweek'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday (numeric)
df.drop(columns=['timestamp'], inplace=True)

# category/numerical
cat_features=df.select_dtypes(object).columns.tolist()

feature1=df.select_dtypes('float64').columns.tolist()
feature2=df.select_dtypes(int).columns.tolist()
num_features=feature1+feature2

# -----Check for null or missing values
df.isnull().sum()

# missing rate calculation per column
# Calculate missing percentage for each column
percent_missing = df.isnull().sum() * 100 / len(df)

# Filter column names where missing percentage is less than 50%
valid_columns = percent_missing[percent_missing < 50].index.tolist()

# Use only the selected columns
df = df[valid_columns]

#df['text'].isnull().sum()
#df = df.dropna(subset=['label'])
#df = df.dropna(subset=['text'])
#df['text'] = df['text'].fillna('')

#------check duplicates
#df = df.drop_duplicates()
df = df.drop_duplicates(subset=['prompt', 'label'])

# -------------------------------------------------
# 2.1. BASIC CLEANING (as an example)
# -------------------------------------------------

'''import re
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters / punctuation (simple approach)
    text = re.sub(r'[^a-zA-ZáéíóúñüÁÉÍÓÚÑÜ\s]', '', text)
    # Trim extra spaces
    text = text.strip()
    return text
data['clean_text'] = data['text'].apply(clean_text) '''
# We’ll use 'clean_text' as our primary input feature


# -----------------------------------------------------------
# 2.2 convert the text into embeddings
# -----------------------------------------------------------

# import torch
# from transformers import DistilBertModel, DistilBertTokenizer
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# model.eval()  # Put the model in eval mode

# def text_to_embedding(text, tokenizer, model):
#     # Tokenize and prepare the inputs
#     # return_tensors="pt" means return PyTorch tensors
#     inputs = tokenizer(
#         text,
#         return_tensors='pt',
#         padding=True,
#         truncation=True,
#         max_length=100
#     )
#     # inputs is a dictionary with keys: ['input_ids','attention_mask'] (DistilBert doesn't use token_type_ids)
    
#     with torch.no_grad():
#         outputs = model(**inputs) 
#         # outputs.last_hidden_state => (batch_size, seq_len, hidden_dim)
    
#     # Use the embedding of the first token [CLS]-like representation
#     last_hidden_states = outputs.last_hidden_state
#     embeddings = last_hidden_states[:, 0, :].squeeze().numpy()
#     return embeddings

# df['embeddings'] = df['prompt'].apply(lambda x: text_to_embedding(x, tokenizer, model))

# X_all = np.stack(df['embeddings'].values)
# y_all = df['label'].values


# -----------------------------------------------------------
# 2.3 Preprocess the dirty / messy text data
# -----------------------------------------------------------
import string
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Prepare stopwords set and lemmatizer once
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

#--------- this is Optional. tfidf later take care of handles tokenization, lowercasing, stopword removal, and punctuation filtering by default.

def preprocess_text(text):
    """
    Basic text preprocessing with NLTK:
    1. Lowercase
    2. Tokenize
    3. Remove punctuation tokens
    4. Remove stopwords
    5. Lemmatize
    Returns a cleaned string.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Tokenize
    tokens = word_tokenize(text)
    
    # 3. Remove punctuation tokens (keep only alphabetic tokens)
    # tokens = [t for t in tokens if t.isalpha()]
    
    # 4. Remove stopwords
    # tokens = [t for t in tokens if t not in stop_words]
    
    # 5. Lemmatize each token
    # tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    # Join tokens back into a single string
    cleaned_text = " ".join(tokens)
    
    return cleaned_text

df['prompt_content_clean'] = df['prompt_content'].apply(preprocess_text)

df_cleaned = df.drop(columns=['id','completion_content','tenure_bin','length_bin','prompt_content_clean'])



# -----------------------------------------------------------
# 3. Data processing and MODELING PIPELINE
# -----------------------------------------------------------
# We'll create a pipeline that does both UNDERSAMPLING + CLASSIFICATION


# -------------------------------------------------
# Optional -  Non-textual data preprocessing
# -------------------------------------------------

# process numerical and categorical data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

text_features = 'prompt_content'
cat_features=df_cleaned.select_dtypes(object).columns.tolist()

feature1=df_cleaned.select_dtypes('float64').columns.tolist()
feature2=df_cleaned.select_dtypes(int).columns.tolist()
num_features=feature1+feature2

from sklearn.feature_extraction.text import TfidfVectorizer
text_transformer = TfidfVectorizer(
    stop_words='english', 
    max_features=1000  # limit vocab size if needed
)

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, text_col),
        ('cat', categorical_transformer, categorical_cols),
        ('num', numeric_transformer, numeric_cols)
    ],
    remainder='drop'
)

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
# import xgboost as xgb

#if not being able use BERT, use tfidf instead but put it into the pipeline
# Optional


undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
# xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# ---- optional using lightgbm
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(n_estimators=100,
max_depth=3,random_state=78,verbose=-1,subsample=0.8,colsample_bytree=0.8,min_child_samples=5)


# Optional if uses tfidf
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    # ('tfidf', text_transformer),
    ('undersample', undersampler),
    ('clf', lgb_model)
])

#  Check performance on training set
y_train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
pr_auc = average_precision_score(y_train, y_train_pred)

# -----------------------------------------------------------
# 4. TRAIN/TEST SPLIT
# -----------------------------------------------------------

df_cleaned = df.drop(columns=['id','completion_content','tenure_bin','length_bin','prompt_content_clean'])
X_all = df_cleaned.drop(columns=['label']) 
y_all = df_cleaned['label'].values

from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=0.2,  # e.g. 80% train, 20% test
    random_state=77,
    stratify=y_all  # helps maintain label distribution
)
# ---- optional to create validation data
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,  # e.g. 80% train, 20% test
    random_state=77,
    stratify=y_train  # helps maintain label distribution
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size:  {X_test.shape[0]}")



# Train the pipeline directly without hyper parameter tuning

# lgb_model.fit(x_train.values,y_train,eval_set=[(x_val.values,y_val)],eval_metric='average_precision',
#               categorical_feature=[23],callbacks=[lgb.early_stopping(10)])
#eval_metrics: ['AUC','ndcg']

pipeline.fit(X_train, y_train)


# -------- Optional 4.1 hyperparameter tuning

from sklearn.model_selection import GridSearchCV

param_grid = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [3, 5],
    'clf__learning_rate': [0.01, 0.1]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring='average_precision',  # e.g., PR-AUC
    cv=3,  # 3-fold cross-validation
    verbose=1,
    n_jobs=-1  # parallelize if desired
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
#print out
grid_search.best_params_
grid_search.best_score_

# -----------------------------------------------------------
# 5. Check performance on test set and threshold selection
# -----------------------------------------------------------

y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
# or
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Evaluate PR-AUC, F1, etc.

from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score


pr_auc = average_precision_score(y_train, y_train_pred)
print(f"Test PR-AUC: {pr_auc:.4f}")

precision, recall, thresholds = precision_recall_curve(y_train, y_train_pred)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best F1: {f1_scores[best_idx]:.4f} at threshold={best_threshold:.4f}")


# And finally, apply that threshold:
y_test_pred_custom = (y_test_probas >= best_threshold).astype(int)
test_f1 = f1_score(y_test, y_test_pred_custom)
print(f"Final F1 (with custom threshold): {test_f1:.4f}")


#----------package the model ---------

import joblib

# Save the pipeline to a file
joblib.dump(pipeline, 'harmful_detection_pipeline.pkl')


#----------load the saved model to use ---------
import joblib
import pandas as pd

# Load the pipeline
loaded_pipeline = joblib.load('harmful_detection_pipeline.pkl')

# Suppose you have new data for inference
new_data = pd.Series([
    "This is a test text that might be harmful.",
    "I love your work, thanks for sharing!"
])

# 1. Get the probability of the positive class
y_scores = pipeline.predict_proba(new_data)[:, 1]

# 2. Apply your custom threshold
custom_threshold = 0.7
y_pred_custom = (y_scores >= custom_threshold).astype(int)


# -----------------------------------------------------------
# 6. POTENTIAL DEPLOYMENT CONSIDERATIONS
# -----------------------------------------------------------
# Example: Outline steps for production (not actual code for deployment, 
# but conceptual approach)

"""
1. Model Packaging:
   - Wrap the model pipeline (TF-IDF + Logistic Regression) into a serialized 
     form (e.g., pickle, joblib).

2. Infrastructure:
   - Deploy as a microservice that takes text input, runs inference, 
     and returns a harmful/not-harmful label or probability.

3. Monitoring:
   - Collect model predictions in real-time to ensure distribution of inputs 
     remains consistent (data drift).
   - Monitor metrics like:
       - Real-time F1, precision, recall (if you have ground truth eventually).
       - Inference latency (for real-time systems).
       - Input text distribution changes over time (data drift).
   - Set up alerts if predictions deviate from expected ranges.

4. Retraining / Model Updates:
   - Periodically retrain or update the model with new data that reflects 
     recent changes in content or adversarial behaviors.
   - A/B test new models in production to evaluate performance improvements 
     or regressions.

5. Explainability / Interpretability:
   - Provide logs or local explanation (e.g., SHAP values or LIME) to see 
     why certain messages are flagged as harmful.

6. Ethical / Safety Considerations:
   - Carefully handle edge cases, false positives (where innocent text 
     is flagged harmful) vs. false negatives (harmful text is missed).
   - Thoroughly evaluate the model for different user demographics, 
     slangs, or languages to reduce bias.
"""



