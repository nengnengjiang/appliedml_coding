#-----------TF–IDF + Undersampling

from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler

# A. TF–IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler

tfidf = TfidfVectorizer(
    stop_words='english', 
    max_features=1000  # limit vocab size if needed
)

# Fit TF–IDF on training set, transform training & test
X_train_tfidf = tfidf.fit_transform(X_train_raw)
X_test_tfidf = tfidf.transform(X_test_raw)

# B. Undersample
undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train_res, y_train_res = undersampler.fit_resample(X_train_tfidf, y_train)

# Convert to dense for PyTorch (WARNING: can be memory-heavy for large data)
X_train_dense = X_train_res.toarray()
X_test_dense = X_test_tfidf.toarray()


#-------Build PyTorch tensors

import torch

# Convert training features & labels
X_train_tensor = torch.tensor(X_train_dense, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_res.values, dtype=torch.long)

# Convert test features & labels
X_test_tensor = torch.tensor(X_test_dense, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)



#---------model defining

import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, num_classes=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # logits
        return x

input_dim = X_train_tensor.shape[1]  # number of features
model = MLP(input_dim=input_dim, hidden_dim=50, num_classes=2)

#----------model training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    
    # Forward pass on the entire training set
    logits = model(X_train_tensor)
    loss = criterion(logits, y_train_tensor)
    
    # Backprop & update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


#=-=---------model eval and threshold selection

from torch.nn.functional import softmax
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

model.eval()
test_probs = []
test_labels = []

with torch.no_grad():
    test_logits = model(X_test_tensor)
    test_probs = softmax(test_logits, dim=1)[:, 1].numpy()  # prob of class=1
    
# PR-AUC
test_pr_auc = average_precision_score(y_test_tensor, test_probs)
print(f"Test PR-AUC: {pr_auc:.4f}")

# Find threshold that maximizes F1 (using test set)
precision, recall, thresholds = precision_recall_curve(y_test_tensor, test_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Best F1 on test: {best_f1:.4f} at threshold={best_threshold:.4f}")
print(f"Precision={precision[best_idx]:.4f}, Recall={recall[best_idx]:.4f}")

# Convert probabilities to 0/1 with custom threshold
test_preds_custom = (test_probs >= best_threshold).astype(int)
final_f1 = f1_score(y_test_tensor, test_preds_custom)
print(f"Final F1 with custom threshold on test: {final_f1:.4f}")





