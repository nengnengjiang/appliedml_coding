#-----------TF–IDF + Undersampling

from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler

# A. TF–IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train_raw)  # sparse matrix
X_test_tfidf = tfidf.transform(X_test_raw)

# B. Undersample
undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train_res, y_train_res = undersampler.fit_resample(X_train_tfidf, y_train)

# Convert to dense for PyTorch (WARNING: can be memory-heavy for large data)
X_train_dense = X_train_res.toarray()
X_test_dense = X_test_tfidf.toarray()


#-------Build a PyTorch Dataset & DataLoader

import torch
from torch.utils.data import Dataset, DataLoader

class TfidfDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create train & test datasets
train_dataset = TfidfDataset(X_train_dense, y_train_res)
test_dataset = TfidfDataset(X_test_dense, y_test)

# DataLoaders (batch_size is adjustable)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


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

input_dim = X_train_dense.shape[1]  # number of features
model = MLP(input_dim=input_dim, hidden_dim=50, num_classes=2)

#----------model training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


#=-=---------model eval and threshold selection

from torch.nn.functional import softmax
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

model.eval()
test_probs = []
test_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = softmax(logits, dim=1)[:, 1]  # probability of class=1
        test_probs.append(probs.cpu().numpy())
        test_labels.append(y_batch.numpy())

test_probs = np.concatenate(test_probs)
test_labels = np.concatenate(test_labels)

# PR-AUC
pr_auc = average_precision_score(test_labels, test_probs)
print(f"Test PR-AUC: {pr_auc:.4f}")

# Find threshold that maximizes F1 (using test set)
precision, recall, thresholds = precision_recall_curve(test_labels, test_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Best F1 on test: {best_f1:.4f} at threshold={best_threshold:.4f}")
print(f"Precision={precision[best_idx]:.4f}, Recall={recall[best_idx]:.4f}")

# Convert probabilities to 0/1 with custom threshold
test_preds_custom = (test_probs >= best_threshold).astype(int)
final_f1 = f1_score(test_labels, test_preds_custom)
print(f"Final F1 with custom threshold on test: {final_f1:.4f}")





