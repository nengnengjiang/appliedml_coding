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
import torch

# -----------------------------------------------------------
# 1. LOAD THE DATA
# -----------------------------------------------------------
data = pd.read_csv('harmful_data.csv')  # Replace with your actual path

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
data.isnull().sum()
# missing rate calculation per column
percent_missing = \
pd.DataFrame(df['text'].isnull().sum() * 100 / len(df)).reset_index()

#only keep the colums which has less than 50%
miss_50_minus=percent_missing.loc[percent_missing.missing_rate<50,'columns'].to_list()
df=df[miss_50_minus]

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




# 2.2 convert the text into embeddings
import torch
from transformers import DistilBertModel, DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model.eval()  # Put the model in eval mode

#2.3 define  FUNCTION TO TRANSFORM TEXT INTO EMBEDDINGS

def text_to_embedding(text, tokenizer, model):
    # Tokenize and prepare the inputs
    # return_tensors="pt" means return PyTorch tensors
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=100
    )
    # inputs is a dictionary with keys: ['input_ids','attention_mask'] (DistilBert doesn't use token_type_ids)
    
    with torch.no_grad():
        outputs = model(**inputs) 
        # outputs.last_hidden_state => (batch_size, seq_len, hidden_dim)
    
    # Use the embedding of the first token [CLS]-like representation
    last_hidden_states = outputs.last_hidden_state
    embeddings = last_hidden_states[:, 0, :].squeeze().numpy()
    return embeddings

df['embeddings'] = df['prompt'].apply(lambda x: text_to_embedding(x, tokenizer, model))

# X_all = np.stack(df['embeddings'].values)
# y_all = df['label'].values

X_all = df.drop(columns=['label']).values
X_all = df['label'].values

# Optional - define which columns go where
text_col = 'prompt / test'
categorical_cols = ['server_name', 'region', 'device', 'ip_country']
numeric_cols = ['user_tenure','hour','timeofday']  # add more if needed

# -----------------------------------------------------------
# 3. TRAIN/TEST SPLIT
# -----------------------------------------------------------

from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=0.2,  # e.g. 80% train, 20% test
    random_state=77,
    stratify=y  # helps maintain label distribution
)
# ---- optional to create validation data
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,  # e.g. 80% train, 20% test
    random_state=77,
    stratify=y  # helps maintain label distribution
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size:  {X_test.shape[0]}")

# -----------------------------------------------------------
# 4. MODELING PIPELINE
# -----------------------------------------------------------
# We'll create a pipeline that does both UNDERSAMPLING + CLASSIFICATION


# -------------------------------------------------
# Optional -  Non-textual data preprocessing
# -------------------------------------------------

# process numerical and categorical data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

'''
from sklearn.feature_extraction.text import TfidfVectorizer
text_transformer = TfidfVectorizer(
    stop_words='english', 
    max_features=1000  # limit vocab size if needed
)
'''
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
import xgboost as xgb

#if not being able use BERT, use tfidf instead but put it into the pipeline
# Optional
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(
    stop_words='english', 
    max_features=1000  # limit vocab size if needed
)

undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# ---- optional using lightgbm
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(n_estimators=100,
,max_depth=3,random_state=78,verbose=-1,subsample=0.8,colsample_bytree=0.8,min_child_samples=5)

pipeline = ImbPipeline([
    ('undersample', undersampler),
    ('clf', xgb_clf)
])

# Optional if uses tfidf
pipeline = ImbPipeline([
    ('tfidf' or 'preprocessor', tfidf or preprocessor),
    ('undersample', undersampler),
    ('clf', xgb_clf)
])

# Train the pipeline
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
    cv=3,
    verbose = 1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
#print out
grid_search.best_params_
grid_search.best_score_

# -----------------------------------------------------------
# 5. EVALUATION and threshold selection
# -----------------------------------------------------------
from sklearn.model_selection import cross_val_predict

# y_pred = best_model.predict_proba(X_test)[:, 1]
y_train_probas = cross_val_predict(
    best_model, 
    X_train, 
    y_train, 
    cv=3, 
    method='predict_proba'
)
y_train_pred = y_train_probas[:, 1]

# Evaluate PR-AUC, F1, etc.

from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

pr_auc = average_precision_score(y_train, y_train_pred)
print(f"Test PR-AUC: {pr_auc:.4f}")

precision, recall, thresholds = precision_recall_curve(y_train, y_train_pred)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best F1: {f1_scores[best_idx]:.4f} at threshold={best_threshold:.4f}")


#Use the best model and train on entire train set again and then evaluate with best threshold on test set
best_model.fit(X_train, y_train)
y_test_probas = best_model.predict_proba(X_test)[:, 1]

# Evaluate final performance with test set
test_pr_auc = average_precision_score(y_test, y_test_probas)

# And finally, apply that threshold:
y_test_pred_custom = (y_test_probas >= best_threshold).astype(int)
test_f1 = f1_score(y_test, y_test_pred_custom)
print(f"Final F1 (with custom threshold): {test_f1:.4f}")



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



