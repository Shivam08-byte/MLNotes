
---

## Section 1: ML / DS / NumPy / Pandas / TensorFlow Questions

### Q1: Difference between NumPy array and Python list?

**Answer:**

| Feature | NumPy Array | Python List |
|---------|-------------|------------|
| **Data Type** | Homogeneous (same data type) | Heterogeneous (mixed data types) |
| **Memory Usage** | Less memory (contiguous block) | More memory (scattered pointers) |
| **Speed** | Faster (optimized C code) | Slower (pure Python) |
| **Convenience** | Need to import NumPy | Built-in, no imports |
| **Operations** | Element-wise operations easily | Requires loops |
| **Size Fixed** | Size fixed after creation | Dynamic size |

**Example:**
```python
# NumPy array
import numpy as np
arr = np.array([1, 2, 3, 4])
arr * 2  # [2, 4, 6, 8] - fast, vectorized

# Python list
lst = [1, 2, 3, 4]
[x * 2 for x in lst]  # Need list comprehension
```

**Key Point:** NumPy arrays are optimized for numerical computing, while lists are general-purpose and flexible.

---

### Q2: What is vectorization and why is it faster?

**Answer:**

Vectorization means performing operations on entire arrays instead of looping through elements one by one.

**Why is it faster?**
- **No Python loop overhead** - Operations run in optimized C code
- **Cache efficiency** - Data is contiguous in memory
- **SIMD (Single Instruction, Multiple Data)** - Processor executes one operation on multiple data points simultaneously
- **Fewer function calls** - One vectorized operation vs multiple loop iterations

**Example:**
```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])

# Vectorized (FAST)
result = arr * 2  # ~1 microsecond

# Loop-based (SLOW)
result = [x * 2 for x in arr]  # ~10 microseconds (10x slower)
```

**Performance Comparison:**
For an array of 1 million elements:
- Vectorized: ~2ms
- Loop-based: ~100ms (50x slower)

---

### Q3: Explain broadcasting with an example.

**Answer:**

Broadcasting allows NumPy to perform operations on arrays of different shapes by automatically expanding smaller arrays to match larger ones.

**Broadcasting Rules:**
1. If arrays have different numbers of dimensions, pad the smaller one with 1s on the left
2. Check if dimensions are compatible (either equal or one is 1)
3. Expand arrays with dimension 1 to match the other array

**Examples:**

```python
import numpy as np

# Example 1: Adding scalar to array
arr = np.array([1, 2, 3, 4])
result = arr + 5  # [6, 7, 8, 9]
# Scalar 5 is broadcasted to [5, 5, 5, 5]

# Example 2: Adding 1D to 2D array
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])  # Shape: (2, 3)
arr_1d = np.array([10, 20, 30])  # Shape: (3,)

result = arr_2d + arr_1d
# arr_1d broadcasted to: [[10, 20, 30],
#                         [10, 20, 30]]
# Result: [[11, 22, 33],
#          [14, 25, 36]]

# Example 3: Column vector addition
col = np.array([[1], [2], [3]])  # Shape: (3, 1)
row = np.array([10, 20, 30])     # Shape: (3,)
result = col + row  # Shape: (3, 3)
```

**Key Point:** Broadcasting avoids unnecessary memory copies and makes code efficient and readable.

---

### Q4: Difference between DataFrame.apply() and vectorized ops.

**Answer:**

| Aspect | apply() | Vectorized Ops |
|--------|---------|----------------|
| **Speed** | Slower (~100-1000ms) | Faster (~1-10ms) |
| **Implementation** | Python loops | Optimized C code |
| **Syntax** | Custom function required | Direct operations |
| **Use Case** | Complex logic per row/column | Simple arithmetic/logic |
| **Flexibility** | Very flexible | Less flexible |

**Example:**
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': [1, 2, 3, 4],
                   'B': [5, 6, 7, 8]})

# Using apply() - SLOWER
result1 = df['A'].apply(lambda x: x * 2 + 10)

# Using vectorized ops - FASTER
result2 = df['A'] * 2 + 10  # 10-100x faster

# apply() is useful when:
result3 = df['A'].apply(lambda x: x**2 if x > 2 else 0)  # Complex logic
```

**Performance (for 1M rows):**
- apply(): ~200ms
- Vectorized: ~5ms (40x faster)

**When to use each:**
- **Vectorized:** Always first choice for mathematical operations
- **apply():** Only when you have complex row/column-specific logic

---

### Q5: How to handle missing values in Pandas?

**Answer:**

**Methods:**

1. **Detect Missing Values:**
```python
df.isnull()          # Returns boolean DataFrame
df.isnull().sum()    # Count missing per column
df.dropna()          # Remove rows with any NaN
df.fillna(0)         # Fill with value
```

2. **Strategies:**

| Strategy | Code | Use Case |
|----------|------|----------|
| **Drop** | `df.dropna()` | Small % of data missing |
| **Fill with constant** | `df.fillna(0)` | When 0 is meaningful |
| **Forward fill** | `df.fillna(method='ffill')` | Time-series data |
| **Fill with mean** | `df.fillna(df.mean())` | Numerical features |
| **Fill with median** | `df.fillna(df.median())` | Outliers present |
| **Interpolate** | `df.interpolate()` | Time-series gaps |

3. **Example:**
```python
# Dataset with missing values
import pandas as pd
df = pd.DataFrame({'A': [1, 2, None, 4],
                   'B': [5, None, None, 8]})

# Strategy 1: Drop rows with any NaN
df.dropna()

# Strategy 2: Fill with column mean
df.fillna(df.mean())

# Strategy 3: Fill only specific columns
df['A'].fillna(df['A'].mean(), inplace=True)

# Strategy 4: Fill with forward fill (for time-series)
df.fillna(method='ffill')
```

**Best Practices:**
- Don't drop too many rows (lose data)
- Use domain knowledge to choose fill value
- For ML: imputation is better than dropping
- Consider using sklearn's SimpleImputer for complex cases

---

### Q6: Difference between join, merge, concat.

**Answer:**

| Operation | Axis | How It Works | Use Case |
|-----------|------|-------------|----------|
| **join()** | Column (axis=1) | Joins on index | Join by index |
| **merge()** | Column (axis=1) | Joins on columns | Join by column values |
| **concat()** | Both (0 or 1) | Stacks DataFrames | Combine multiple DFs |

**Examples:**

```python
import pandas as pd

df1 = pd.DataFrame({'key': ['A', 'B', 'C'],
                    'val1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'C'],
                    'val2': [10, 20, 30]})

# join() - on index
df1.set_index('key').join(df2.set_index('key'))
# Joins rows with same index

# merge() - on column
result = pd.merge(df1, df2, on='key', how='inner')
# Inner join on 'key' column

# concat() - stack them
pd.concat([df1, df2], axis=0)  # Stack rows
pd.concat([df1, df2], axis=1)  # Stack columns
```

**When to use:**
- **join():** Simple index-based joins
- **merge():** Join by specific columns (SQL-like)
- **concat():** Combine multiple DataFrames, different structures

---

### Q7: What is a tensor? Explain in simple words.

**Answer:**

A **tensor** is a multi-dimensional array of numbers.

**Dimensions:**
- **0-D Tensor (Scalar):** Single number: `5`
- **1-D Tensor (Vector):** List of numbers: `[1, 2, 3, 4]`
- **2-D Tensor (Matrix):** 2D array: `[[1, 2], [3, 4]]`
- **3-D Tensor:** 3D array: `[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]`
- **4-D Tensor and beyond:** Images (H×W×C), batches of images (B×H×W×C)

**Real-world Examples:**
- **Scalar:** Temperature value `25.5`
- **Vector:** Image pixel values `[200, 150, 100]` (RGB)
- **Matrix:** Grayscale image `28×28` pixels
- **3-D Tensor:** Color image `256×256×3` (height × width × color channels)
- **4-D Tensor:** Batch of 32 images `32×256×256×3` (batch × height × width × channels)

**In Code:**
```python
import tensorflow as tf
import numpy as np

# 0-D Tensor (scalar)
t0 = tf.constant(5)

# 1-D Tensor (vector)
t1 = tf.constant([1, 2, 3, 4])

# 2-D Tensor (matrix)
t2 = tf.constant([[1, 2], [3, 4]])

# 3-D Tensor (batch of matrices)
t3 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# 4-D Tensor (batch of images)
t4 = tf.random.normal((32, 28, 28, 3))  # 32 images, 28×28 pixels, 3 color channels
```

**Key Point:** Tensors are the fundamental data structure in deep learning frameworks like TensorFlow and PyTorch.

---

### Q8: What is model.fit() in TensorFlow?

**Answer:**

`model.fit()` is the method that trains a neural network on training data.

**What it does:**
1. Feeds training data in batches to the model
2. Computes predictions
3. Calculates loss (error)
4. Computes gradients using backpropagation
5. Updates weights using optimizer
6. Repeats for all epochs

**Syntax:**
```python
model.fit(
    x=X_train,              # Input features
    y=y_train,              # Target labels
    batch_size=32,          # Samples per batch
    epochs=10,              # Number of passes through data
    validation_data=(X_val, y_val),  # Validation set
    callbacks=[],           # Early stopping, etc.
    verbose=1               # 0=silent, 1=progress bar
)
```

**Example:**
```python
import tensorflow as tf
from tensorflow import keras

# Build model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_val, y_val),
    verbose=1
)
```

**Returns:**
- `History` object containing loss and metrics for each epoch
- Allows plotting training progress

**Key Parameters:**
- **epochs:** More epochs = longer training, potentially better accuracy (but overfitting risk)
- **batch_size:** Larger batches = faster training but less accurate updates
- **validation_data:** Monitor generalization during training

---

### Q9: What is the difference between TensorFlow and PyTorch?

**Answer:**

| Aspect | TensorFlow | PyTorch |
|--------|-----------|---------|
| **Developed by** | Google | Facebook (Meta) |
| **Graph Type** | Static graph (define then run) | Dynamic graph (define by run) |
| **Ease of Learning** | Steeper learning curve | Intuitive, Pythonic |
| **Production** | Better for production | Growing production support |
| **Debugging** | Harder (static graphs) | Easier (Python debugging) |
| **Research** | Slower to prototype | Faster to prototype |
| **Performance** | Highly optimized | Good performance |
| **Community** | Large, established | Growing rapidly |
| **Mobile Deployment** | TensorFlow Lite (excellent) | TorchMobile (decent) |

**Code Comparison:**

**TensorFlow:**
```python
import tensorflow as tf

# Static graph
@tf.function
def forward(x):
    return tf.nn.relu(tf.matmul(x, W) + b)

output = forward(input_tensor)
```

**PyTorch:**
```python
import torch

# Dynamic graph
def forward(x):
    return torch.relu(torch.matmul(x, W) + b)

output = forward(input_tensor)
```

**When to use:**
- **TensorFlow:** Production systems, mobile apps, when you need maximum optimization
- **PyTorch:** Research, prototyping, learning, academic projects

---

### Q10: Explain GPU acceleration in PyTorch.

**Answer:**

GPU acceleration means running computations on Graphics Processing Unit (GPU) instead of CPU for massive speedup.

**Why GPU is faster:**
- **Parallel Processing:** GPUs have thousands of cores (vs CPU's ~8-16)
- **Designed for matrices:** GPUs excel at matrix operations (deep learning's core operation)
- **Memory bandwidth:** Much higher data throughput than CPU

**Performance:**
- CPU: ~10-50 GFLOPS (billion floating-point operations)
- GPU (NVIDIA RTX 3080): ~30,000 GFLOPS (3000x faster!)

**How to use in PyTorch:**

```python
import torch

# Check if GPU available
print(torch.cuda.is_available())  # True if NVIDIA GPU present
print(torch.cuda.get_device_name(0))  # GPU name

# Move tensors to GPU
x = torch.tensor([1.0, 2.0, 3.0])
x_gpu = x.to('cuda')  # or x.cuda()

# Move model to GPU
model = MyModel()
model = model.to('cuda')

# Forward pass on GPU
output = model(x_gpu)

# Alternatively, create tensor directly on GPU
y_gpu = torch.tensor([1.0, 2.0, 3.0], device='cuda')

# Move back to CPU for visualization
result_cpu = output.cpu().numpy()
```

**Practical Example:**
```python
import torch
import time

# Create large tensor
x = torch.randn(10000, 10000)
y = torch.randn(10000, 10000)

# CPU computation
start = time.time()
z_cpu = torch.matmul(x, y)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.3f}s")

# GPU computation
x_gpu = x.cuda()
y_gpu = y.cuda()
start = time.time()
z_gpu = torch.matmul(x_gpu, y_gpu)
gpu_time = time.time() - start
print(f"GPU time: {gpu_time:.3f}s")
# GPU is ~100-1000x faster
```

**Best Practices:**
- Use GPU only for large matrices (overhead of data transfer)
- Keep data on GPU as long as possible
- Use batching to maximize GPU utilization
- Monitor with `nvidia-smi` command

---

### Q11: What is gradient descent?

**Answer:**

Gradient descent is an optimization algorithm that iteratively updates model weights to minimize loss (error).

**Intuition:**
Imagine you're on a mountain and want to reach the valley (minimum loss). Gradient descent tells you which direction to walk (steepest downhill) and how big a step to take.

**Algorithm:**
```
1. Initialize weights randomly
2. Compute loss on training data
3. Compute gradient (direction of steepest increase in loss)
4. Update weights: w = w - learning_rate × gradient
5. Repeat until convergence
```

**Mathematics:**
```
gradient = dL/dw (derivative of loss with respect to weight)
new_weight = old_weight - learning_rate × gradient
```

**Visual Example:**
```
Loss
  ^
  |     ***
  |    *   *
  |   *     *  <- Current position (high loss)
  |  *       *
  | *         *  <- Moving down (lower loss)
  |*           * <- Valley (minimum loss - goal!)
  +-------------------> weight
     Gradient descent moves down
```

**Types of Gradient Descent:**

| Type | Data Used | Speed | Stability |
|------|-----------|-------|-----------|
| **Batch GD** | All data | Slow | Stable |
| **SGD** | One sample | Fast | Unstable |
| **Mini-batch** | Small batch | Medium | Balanced |

**Code Example:**
```python
import torch
import torch.nn as nn

# Simple model
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # lr = learning rate
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(100):
    # Forward pass
    predictions = model(X_train)
    loss = loss_fn(predictions, y_train)
    
    # Backward pass (compute gradients)
    optimizer.zero_grad()  # Reset gradients
    loss.backward()        # Compute gradients
    
    # Update weights (gradient descent step)
    optimizer.step()       # Update: w = w - lr × gradient
    
    print(f"Epoch {epoch}, Loss: {loss.item()}")
```

**Key Points:**
- **Learning rate:** Too high = overshooting; too low = slow convergence
- **Iterations needed:** More complex problems need more iterations
- **Local minima:** May get stuck in local minimum (local optimum)

---

### Q12: Explain overfitting and regularization.

**Answer:**

**Overfitting:**
Model memorizes training data including noise, performs well on training but poorly on new data.

**Visualization:**
```
Accuracy
   ^
   |        Training accuracy (high!)
   |       /
   |      /
   |     /
   |    /
   |   /
   | /
   |/  Test accuracy (low!) - OVERFITTING OCCURS
   +---> Model Complexity
      Sweet spot
```

**Causes of Overfitting:**
- Model too complex (too many parameters)
- Training data too small
- Training for too many epochs
- No regularization

**Regularization Techniques:**

1. **L1/L2 Regularization:**
```python
# L2 (Ridge) Regularization
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
# weight_decay = L2 regularization strength

# Mathematically: Loss = MSE + lambda * sum(w²)
```

2. **Dropout:**
```python
model = nn.Sequential(
    nn.Linear(10, 128),
    nn.ReLU(),
    nn.Dropout(0.5),  # Randomly drop 50% neurons during training
    nn.Linear(128, 1)
)
```

3. **Early Stopping:**
```python
# Stop training when validation loss stops improving
best_val_loss = float('inf')
patience = 5
epochs_without_improvement = 0

for epoch in range(1000):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # Save model
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break  # Stop training
```

4. **Data Augmentation:**
```python
# Increase training data variety
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2)
])
```

5. **Cross-Validation:**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)  # 5-fold CV
print(f"Average accuracy: {scores.mean()}")
```

**How to Detect Overfitting:**
- Large gap between training and validation accuracy
- Training loss keeps decreasing but validation loss increases

---

### Q13: Explain train-test split + why do we need validation set?

**Answer:**

**Data Splitting:**
```
Original Dataset
    |
    +-- Training Set (60-70%) --> Train model
    |
    +-- Validation Set (10-20%) --> Tune hyperparameters, select best model
    |
    +-- Test Set (10-20%) --> Final evaluation (unseen data)
```

**Example:**
```python
from sklearn.model_selection import train_test_split

# Split into train (60%) and temp (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Split temp into validation (50% of temp = 20% of total) and test (20%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
```

**Why we need Validation Set:**

| Purpose | Set Used | Why |
|---------|----------|-----|
| **Model Learning** | Training | Train on this data |
| **Hyperparameter Tuning** | Validation | Choose learning rate, batch size, etc. |
| **Performance Estimate** | Test | Final unbiased evaluation |

**Without validation set:**
- You might pick hyperparameters that overfit to test set
- Final test accuracy would be unreliable

**Flow:**
```
1. Train model on training set
   ↓
2. Evaluate multiple hyperparameters on validation set
   ↓
3. Pick best hyperparameters
   ↓
4. Report final accuracy on test set (NEW, UNSEEN DATA)
```

**Code Example:**
```python
# Training loop with validation
best_model = None
best_val_acc = 0

for lr in [0.001, 0.01, 0.1]:
    for batch_size in [16, 32, 64]:
        model = train(X_train, y_train, lr=lr, batch_size=batch_size)
        val_acc = evaluate(model, X_val, y_val)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

# Final evaluation on unseen test set
test_acc = evaluate(best_model, X_test, y_test)
print(f"Test Accuracy: {test_acc}")
```

**Best Practices:**
- Always use a separate test set (never touch during training/tuning)
- Use stratified split for imbalanced classes
- Keep test set representative of real-world data

---

### Q14: What is cross-validation?

**Answer:**

Cross-validation is a technique to evaluate model performance using multiple train-test splits.

**Why?**
- Single train-test split may be biased
- Different random splits give different results
- Cross-validation gives average performance across splits

**K-Fold Cross-Validation (most common):**
```
Original Dataset
    |
    +-- Fold 1: Train on folds 2-5, test on fold 1
    +-- Fold 2: Train on folds 1,3-5, test on fold 2
    +-- Fold 3: Train on folds 1-2,4-5, test on fold 3
    +-- Fold 4: Train on folds 1-3,5, test on fold 4
    +-- Fold 5: Train on folds 1-4, test on fold 5
    |
    +-- Average accuracy across all folds
```

**Code Example:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Scores: {scores}")  # [0.92, 0.94, 0.91, 0.95, 0.93]
print(f"Average: {scores.mean():.3f}")  # 0.930
print(f"Std Dev: {scores.std():.3f}")   # 0.016 (low std = stable)
```

**Types:**

1. **K-Fold:** Divide into k equal parts
2. **Stratified K-Fold:** Maintains class distribution (for imbalanced data)
3. **Leave-One-Out CV:** Each sample is test set once (slow)
4. **Time-Series CV:** For sequential data, test always after training window

```python
# Stratified K-Fold for imbalanced data
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate
```

**When to use:**
- Small datasets (makes best use of limited data)
- Want robust performance estimate
- Hyperparameter tuning (tune on CV scores)

**Disadvantage:**
- Slower (trains k times)
- May overfit to training set

---

### Q15: Evaluation metrics: accuracy, precision, recall, F1.

**Answer:**

**Confusion Matrix (Classification):**
```
                Predicted
              Positive  Negative
Actual Positive   TP      FN
       Negative   FP      TN

TP = True Positive (correctly predicted positive)
FP = False Positive (incorrectly predicted positive)
FN = False Negative (incorrectly predicted positive as negative)
TN = True Negative (correctly predicted negative)
```

**Metrics:**

| Metric | Formula | Interpretation | Use Case |
|--------|---------|-----------------|----------|
| **Accuracy** | (TP+TN)/(TP+FP+FN+TN) | Overall correctness | Balanced classes |
| **Precision** | TP/(TP+FP) | Of positive predictions, how many correct? | When false positives are costly |
| **Recall** | TP/(TP+FN) | Of actual positives, how many found? | When false negatives are costly |
| **F1** | 2×(Precision×Recall)/(Precision+Recall) | Harmonic mean of precision & recall | Imbalanced classes |

**Real-world Examples:**

**Email Spam Detection:**
- Precision important: Don't want important emails marked as spam (FP is bad)
- Recall important: Don't want spam in inbox (FN is bad)
- Use F1 score

**Medical Diagnosis (disease detection):**
- Recall most important: Don't miss positive cases (FN is very bad)
- Precision less critical: False alarm is ok, follow-up tests confirm

**Credit Card Fraud:**
- Precision important: Don't block legitimate transactions (FP)
- Recall important: Don't miss fraud (FN)
- Use both or F1

**Code Example:**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]

accuracy = accuracy_score(y_true, y_pred)        # 0.75
precision = precision_score(y_true, y_pred)      # 0.67
recall = recall_score(y_true, y_pred)            # 0.67
f1 = f1_score(y_true, y_pred)                    # 0.67

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
# [[3, 1],   -> True Negatives=3, False Positives=1
#  [1, 3]]   -> False Negatives=1, True Positives=3
```

**Precision-Recall Tradeoff:**
```
Threshold high → Fewer predictions, high precision, low recall
Threshold low → More predictions, low precision, high recall
```

**Best Practice:**
- Use appropriate metric for business problem
- For imbalanced data: use F1, not accuracy
- Report all metrics for transparency

---

### Q16: What are embeddings?

**Answer:**

**Embeddings** are dense vector representations of categorical or semantic data in lower-dimensional space.

**Why embeddings?**
- Convert words, entities, or items to numerical vectors
- Similar items have similar vectors (close in vector space)
- Can use vectors in ML models

**Types:**

1. **Word Embeddings (Word2Vec, GloVe, FastText):**
```
Word → Vector (e.g., 300-dimensional)

"king" → [0.2, 0.5, -0.3, ..., 0.1]
"queen" → [0.3, 0.6, -0.2, ..., 0.2]
"man" → [0.1, 0.4, -0.5, ..., 0.0]

king - man ≈ queen - woman  (Vector algebra works!)
```

2. **User/Item Embeddings (Recommendation Systems):**
```
User ID → User embedding
Item ID → Item embedding

Predict: User vector · Item vector = Recommendation score
```

3. **Sentence/Document Embeddings (BERT, GPT):**
```
Sentence → Dense vector (e.g., 768-dimensional)
```

**Example - Word2Vec in Gensim:**
```python
from gensim.models import Word2Vec

# Train on sentences
sentences = [
    ['king', 'rules', 'kingdom'],
    ['queen', 'rules', 'kingdom'],
    ['man', 'walks', 'street'],
    ['woman', 'walks', 'street']
]

model = Word2Vec(sentences, vector_size=100, window=5)

# Get vector for a word
king_vector = model.wv['king']  # Shape: (100,)

# Find similar words
similar = model.wv.most_similar('king', topn=5)
# [('queen', 0.85), ('prince', 0.78), ...]

# Vector operations
result = model.wv['king'] - model.wv['man'] + model.wv['woman']
closest = model.wv.similar_by_vector(result, topn=1)
# Likely to be 'queen'!
```

**Embedding Size:**
- Smaller (50): Fast, less memory, less expressive
- Larger (300, 768): Slower, more memory, more expressive
- Rule of thumb: 50-300 for most applications

**Key Properties:**
- **Distributed representation:** Meaning spread across dimensions
- **Similarity preserved:** Similar words have similar vectors
- **Composability:** Vector operations make semantic sense

---

### Q17: Why do we use normalization/standardization?

**Answer:**

**Normalization/Standardization** scales features to a standard range or distribution.

**Why?**
1. **Features on different scales:** Age (0-100) vs Income (0-1,000,000)
2. **ML algorithms perform better:** Especially distance-based and gradient descent-based
3. **Faster convergence:** Training takes fewer iterations
4. **Prevents overflow:** Avoids numerical issues

**Impact:**
```
Without scaling: Income dominates age in distance calculations
                 Distance ≈ (Income_diff)² + (Age_diff)²
                 Distance ≈ large number + small number ≈ large number
                 Age is ignored!

With scaling:   All features equally important
                 Distance includes contribution from all features
```

**Normalization vs Standardization:**

| Type | Formula | Range | When |
|------|---------|-------|------|
| **Normalization (Min-Max)** | (x - min)/(max - min) | [0, 1] | Bounded range needed |
| **Standardization (Z-score)** | (x - mean)/std | Center at 0, std=1 | Normal distribution assumed |

**Code Examples:**

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

X = np.array([[1, 2], [3, 4], [5, 6]])

# Normalization (Min-Max Scaling)
scaler_norm = MinMaxScaler()
X_normalized = scaler_norm.fit_transform(X)
# X_normalized = [[0, 0], [0.5, 0.5], [1, 1]]

# Standardization (Z-score)
scaler_std = StandardScaler()
X_standardized = scaler_std.fit_transform(X)
# X_standardized ≈ [[-1.22, -1.22], [0, 0], [1.22, 1.22]]

# For new data during prediction
new_X = np.array([[2, 3]])
new_X_transformed = scaler_norm.transform(new_X)
# Use SAME scaler fitted on training data!
```

**When to use:**

| Algorithm | Needs Scaling? | Why |
|-----------|---|---|
| **Linear Regression** | Yes | Coefficients affected by scale |
| **Logistic Regression** | Yes | Gradient descent convergence |
| **Neural Networks** | Yes | Faster convergence, stability |
| **SVM** | Yes | Distance-based, sensitive to scale |
| **KNN** | Yes | Distance calculations |
| **Decision Trees** | No | Scale-invariant |
| **Random Forest** | No | Scale-invariant |

**Important:** Fit scaler on training data only, then transform test data

```python
# CORRECT
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!

# WRONG
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)  # Different scaler!
```

---

### Q18: Explain softmax.

**Answer:**

**Softmax** is an activation function that converts raw scores (logits) into probability distribution.

**Formula:**
```
softmax(x_i) = e^(x_i) / Σ(e^(x_j))

For vector [2, 1, 0.1]:
e^2 ≈ 7.39
e^1 ≈ 2.72
e^0.1 ≈ 1.11
Sum ≈ 11.22

softmax = [7.39/11.22, 2.72/11.22, 1.11/11.22]
        = [0.659, 0.242, 0.099]  (probabilities sum to 1)
```

**Why softmax?**
1. **Converts to probabilities:** Output sums to 1
2. **Differentiable:** Good for backpropagation
3. **Emphasizes larger values:** Exponential function creates distinction
4. **Multi-class classification:** Standard choice

**Use Cases:**

1. **Multi-class Classification:**
```python
import torch
import torch.nn as nn

# 3 classes
logits = torch.tensor([[2.0, 1.0, 0.1]])
softmax = nn.Softmax(dim=1)
probabilities = softmax(logits)
# Output: [[0.659, 0.242, 0.099]]

# Class prediction
predicted_class = torch.argmax(probabilities, dim=1)  # Class 0
```

2. **In Neural Network:**
```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)  # Multi-class output
)
```

3. **With Cross-Entropy Loss:**
```python
# Softmax + Cross-Entropy is standard combination
logits = model(X)  # Raw scores
loss_fn = nn.CrossEntropyLoss()  # Applies softmax internally + cross-entropy
loss = loss_fn(logits, y_true)

# Alternative (manual)
probabilities = nn.Softmax(dim=1)(logits)
loss = nn.NLLLoss()(torch.log(probabilities), y_true)
```

**Softmax vs Sigmoid:**

| Aspect | Softmax | Sigmoid |
|--------|---------|---------|
| **Classes** | Multi-class (>2) | Binary (2 classes) |
| **Output** | Probability distribution | Single probability |
| **Sum of outputs** | Always 1 | Each output independent |

**Visualization:**
```
Raw Scores (Logits):  [2, 1, 0.1]
                           ↓ softmax
Probabilities:        [0.659, 0.242, 0.099]
                           ↓
Prediction:           Class 0 (highest prob)
```

**Key Point:** Softmax is almost always used for multi-class classification's final layer.

---

### Q19: What is a confusion matrix?

**Answer:**

**Confusion Matrix** is a table showing actual vs predicted classifications.

**Structure (Binary Classification):**
```
                Predicted
              Positive  Negative
Actual Pos      TP        FN       (FN: missed positive)
       Neg      FP        TN       (FP: false alarm)
```

**Example:**
```
Actual positive: 100 samples
Actual negative: 200 samples

                Predicted Positive  Predicted Negative
Actual Positive        80 (TP)              20 (FN)
Actual Negative        10 (FP)              190 (TN)
```

**Derived Metrics:**
- Accuracy = (TP + TN) / Total = (80 + 190) / 300 = 0.90
- Precision = TP / (TP + FP) = 80 / 90 = 0.89
- Recall = TP / (TP + FN) = 80 / 100 = 0.80
- F1 = 2 × (0.89 × 0.80) / (0.89 + 0.80) = 0.84

**Multi-Class Example:**
```
                Predicted
              Dog  Cat  Bird
Actual Dog   [45   5    0]
       Cat   [3   42   5]
       Bird  [0    2   48]
```

**Code Example:**
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1, 1, 1, 2]

cm = confusion_matrix(y_true, y_pred)
print(cm)
# [[2 0 0]
#  [1 1 1]
#  [0 1 2]]

# Visualize
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Dog', 'Cat', 'Bird'])
disp.plot()
plt.show()
```

**Interpretation:**
- Diagonal (TP, TN): Correct predictions
- Off-diagonal: Errors (FP, FN)
- Dark diagonal = good model

**Use For:**
- Identifying which classes model confuses
- Computing metrics (accuracy, precision, recall, F1)
- Finding class-specific performance
- Detecting class imbalance handling

---

### Q20: What is batch size vs epoch?

**Answer:**

**Batch Size:**
- Number of samples processed before updating weights
- Single forward + backward pass = one weight update

**Epoch:**
- One complete pass through entire training dataset

**Example:**
```
Training data: 1000 samples
Batch size: 100 samples

1 Epoch = 10 iterations (1000 / 100)
          (each iteration processes 100 samples)

10 Epochs = 100 iterations total
```

**Impact:**

| Aspect | Small Batch | Large Batch |
|--------|-------------|------------|
| **Speed per epoch** | Slow (many updates) | Fast (few updates) |
| **Memory** | Low | High |
| **Stability** | Noisy (might escape local minima) | Smooth |
| **Convergence** | May require more epochs | Faster convergence |
| **Generalization** | Often better | Sometimes worse |

**Visualization:**
```
Large batch size:  Smooth but may miss optimum
                  Loss curve: gentle descent
                   
Small batch size: Noisy but explores more
                  Loss curve: jagged but explores well
```

**Code Example:**
```python
import torch
from torch.utils.data import DataLoader

# Dataset: 10000 samples
dataset = torch.randn(10000, 10)

# Batch size = 32
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 1 Epoch = 10000 / 32 ≈ 313 iterations
for epoch in range(10):  # 10 epochs
    for batch_idx, batch in enumerate(dataloader):
        # batch contains 32 samples
        # Process batch, compute loss, backprop, update weights
        pass
    # One epoch complete
    print(f"Epoch {epoch} complete")
```

**Common Batch Sizes:**
- 32, 64, 128, 256: Standard for most models
- 1: Stochastic Gradient Descent (SGD)
- Entire dataset: Batch Gradient Descent

**Epoch Selection:**
- Too few epochs: Underfitting
- Too many epochs: Overfitting
- Use early stopping to find optimal

**Tips:**
- Larger batch: Use more epochs
- Smaller batch: May need fewer epochs
- Monitor validation loss to detect overfitting

---

## Section 2: Quick ML FAQs Sheet (20-Minute Revision)

### Supervised vs Unsupervised learning.

**Supervised Learning:**
- Labeled data (input + target output)
- Task: Learn mapping from input to output
- Examples: Regression, Classification
- Use: Prediction, forecasting

**Unsupervised Learning:**
- Unlabeled data (input only)
- Task: Find hidden patterns/structure
- Examples: Clustering, Dimensionality reduction
- Use: Exploration, grouping

---

### Regression vs Classification (quick examples).

**Regression:**
- Continuous output (real numbers)
- Examples: House price, temperature, stock price
- Metrics: MSE, RMSE, R²

**Classification:**
- Discrete output (categories)
- Examples: Email spam (yes/no), sentiment (positive/negative/neutral)
- Metrics: Accuracy, Precision, Recall

---

### Feature engineering basics.

Creating/selecting relevant features improves model performance.

**Techniques:**
- Feature scaling (normalization, standardization)
- Feature selection (remove irrelevant features)
- Feature creation (combine existing features)
- Encoding (categorical to numerical)

---

### Handling missing values.

**Strategies:**
- Drop rows with missing values
- Fill with mean/median/mode
- Forward fill (time-series)
- Imputation (predict missing values)

---

### Normalization vs Standardization.

- **Normalization:** Scale to [0, 1]
- **Standardization:** Scale to mean=0, std=1

---

### Train/Test/Validation split.

- **Training:** 60-70% (train model)
- **Validation:** 10-20% (tune hyperparameters)
- **Testing:** 10-20% (final evaluation)

---

### Bias-Variance tradeoff.

- **High Bias:** Underfitting (model too simple)
- **High Variance:** Overfitting (model too complex)
- **Goal:** Balance both for generalization

---

### Overfitting and how to avoid it.

**Prevention:**
- More training data
- Regularization (L1/L2, dropout)
- Early stopping
- Simpler model
- Cross-validation

---

### Evaluation metrics for classifiers.

Accuracy, Precision, Recall, F1, AUC-ROC

---

### One-hot encoding vs Label encoding.

- **One-hot:** Categorical to binary vectors (for nominal data)
- **Label:** Categorical to integers (for ordinal data)

---

### When to use Random Forest?

Non-linear relationships, feature importance needed, robust to outliers

---

### When to use Logistic Regression?

Interpretability needed, fast, linear relationships, probability output

---

### Purpose of loss function.

Quantifies prediction error, guides optimization

---

### Purpose of activation functions.

Introduce non-linearity, enable complex function learning

---

### Gradient descent (intuition only).

Iteratively move in direction of steepest descent to minimize loss

---

### Learning rate & its impact.

- Too high: Overshooting, divergence
- Too low: Slow convergence, local minima
- Optimal: Smooth convergence

---

### Cross-validation.

Multiple train-test splits for robust performance estimate

---

### Embeddings (simple).

Dense vector representation of categorical/semantic data

---

## Section 3: AI-Specific Expected FAQs Questions

### Q1: Explain how your GenAI summarizer works end-to-end.

**Answer:**

**Pipeline:**
```
1. Input Text
   ↓
2. Preprocessing (clean, remove special chars)
   ↓
3. Text Chunking (break into 512-token chunks)
   ↓
4. Embedding Generation (convert chunks to vectors)
   ↓
5. Vector Storage (store in Pinecone/FAISS)
   ↓
6. Prompt Construction (build context-aware prompt)
   ↓
7. LLM Call (send prompt to GPT/Claude)
   ↓
8. Structured Output (parse LLM response)
   ↓
9. FastAPI Endpoint (serve via REST API)
   ↓
10. AKS Deployment (containerized on Kubernetes)
```

**Details:**
- **Input:** Long document (10,000+ tokens)
- **Chunking:** Break into overlapping 512-token chunks with 50-token overlap
- **Embeddings:** Convert using OpenAI embeddings API (1536-dim vectors)
- **Storage:** Index in vector DB for semantic search
- **Prompt:** "Summarize in 3-5 sentences: {retrieved_context}"
- **Output:** Structured JSON with summary, key points, sentiment
- **Deployment:** Docker container on AKS with GPU support

---

### Q2: Difference between prompting, RAG, and fine-tuning.

**Answer:**

| Aspect | Prompting | RAG | Fine-tuning |
|--------|-----------|-----|------------|
| **Definition** | Direct instruction in prompt | Retrieval + generation | Train on custom data |
| **Cost** | Low (API calls) | Medium (retrieval + API) | High (compute, data) |
| **Latency** | Low | Medium (retrieval adds time) | Low (inference) |
| **Use Case** | General tasks | Domain-specific, latest data | Specific behavior/style |
| **Example** | "Summarize this text" | "Answer based on docs" | "Respond like sales rep" |

**When to use:**
- **Prompting:** Quick tasks, general knowledge
- **RAG:** Private docs, updated info needed, accuracy critical
- **Fine-tuning:** Specific domain, consistent behavior, cost-effective at scale

---

### Q3: What are LLM tokens?

**Answer:**

**Token:** Smallest unit LLM processes. Usually 1 word ≈ 1.3 tokens.

**Examples:**
- "hello" → 1 token
- "Hello, world!" → 3 tokens
- "summarization" → 2 tokens

**Cost implications:**
- GPT-4: $0.01 per 1K input tokens, $0.03 per 1K output tokens
- Longer text = more tokens = higher cost

---

### Q4: What is context window? How does it affect summarization?

**Answer:**

**Context Window:** Maximum tokens model can process in one call.

**Examples:**
- GPT-3.5: 4K tokens
- GPT-4: 8K or 32K tokens
- Claude 3: 100K tokens
- Llama 2: 4K tokens

**Impact on Summarization:**
- **Small window (4K):** Can't process long documents, need chunking
- **Large window (32K+):** Can process entire document at once, better coherence
- **Chunking trade-off:** Chunks don't interact, may miss connections

---

### Q5: How do you chunk long text for LLMs?

**Answer:**

**Chunking Strategies:**

1. **Fixed Size Chunks:**
```python
chunk_size = 512  # tokens
overlap = 50      # tokens

chunks = []
for i in range(0, len(tokens), chunk_size - overlap):
    chunk = tokens[i:i + chunk_size]
    chunks.append(chunk)
```

2. **Semantic Chunking (by sentences/paragraphs):**
```python
chunks = text.split('\n\n')  # Split by paragraphs
# Or use regex for sentences
```

3. **Recursive Chunking (Langchain):**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
chunks = splitter.split_text(text)
```

**Best Practices:**
- Overlap helps with context preservation
- Balance between chunk size and coherence
- Consider token limits when encoding

---

### Q6: Explain embeddings with a real example.

**Answer:**

**Embeddings:** Dense vector representation enabling semantic search.

**Real Example - Document Search:**
```
User Query: "How to train ML models?"
Query Embedding: [0.2, -0.5, 0.8, ..., 0.1]  (1536-dim)

Documents:
"ML training basics" → [0.21, -0.51, 0.79, ..., 0.11]  (Similarity: 0.99)
"Cooking recipes" → [0.8, 0.1, -0.2, ..., 0.7]  (Similarity: 0.10)
"Model optimization" → [0.19, -0.48, 0.82, ..., 0.12]  (Similarity: 0.98)

Top Results: ML training basics, Model optimization
```

**Why Useful:**
- Semantic similarity (not keyword matching)
- Enables vector search in databases
- Powers recommendation systems

---

### Q7: What is vector search?

**Answer:**

**Vector Search:** Find similar items by searching in vector space using distance/similarity metrics.

**Process:**
```
1. Convert query to embedding
2. Search vector database for nearest neighbors
3. Return results sorted by similarity
```

**Distance Metrics:**
- **Euclidean:** Straight-line distance (L2)
- **Cosine:** Angle between vectors (best for text)
- **Manhattan:** Grid distance (L1)

**Databases:**
- Pinecone (fully managed)
- FAISS (open-source, Facebook)
- Weaviate (open-source)
- Elasticsearch (with vector plugin)

---

### Q8: How do you evaluate LLM output?

**Answer:**

**Automated Metrics:**
- **BLEU:** Compares with reference (precision-based)
- **ROUGE:** Overlap-based (similar to BLEU)
- **BERTScore:** Semantic similarity using embeddings

**Human Evaluation:**
- Relevance (0-5 scale)
- Coherence (logical flow)
- Factual accuracy
- Tone/style appropriateness

**Code Example:**
```python
from rouge import Rouge

reference = "Machine learning is a subset of AI"
prediction = "ML is part of AI"

rouge = Rouge()
scores = rouge.get_scores(prediction, reference)
# {'rouge1': {'f': 0.75}, 'rouge2': {'f': 0.5}, ...}
```

---

### Q9: What is an agent in Agentic AI?

**Answer:**

**Agent:** AI system that perceives environment, reasons, and takes actions autonomously.

**Components:**
1. **Perception:** Gather information (APIs, tools)
2. **Reasoning:** LLM decides next action
3. **Action:** Execute tool/API call
4. **Feedback Loop:** Update and iterate

**Example - Customer Service Agent:**
```
User: "My order hasn't arrived"
  ↓
Agent: Perceives question via LLM
  ↓
Agent: Reasons: "Need to check order status"
  ↓
Agent: Action: Calls ORDER_API(customer_id)
  ↓
Agent: Receives: Order status = "Shipped"
  ↓
Agent: Reasons: "Need tracking info"
  ↓
Agent: Action: Calls TRACKING_API(order_id)
  ↓
Agent: Response: "Your package is in transit, arrives tomorrow"
```

---

### Q10: Explain tool calling/function calling.

**Answer:**

**Function Calling:** LLM decides when/what external tools to call.

**LLM Output (JSON):**
```json
{
  "function_name": "get_weather",
  "parameters": {
    "city": "Mumbai",
    "units": "celsius"
  }
}
```

**System Action:**
```python
if function_name == "get_weather":
    result = get_weather(city, units)
    return result to LLM
```

**Example - Travel Booking Agent:**
```
User: "Book a flight to NYC for tomorrow"
  ↓
LLM: Call search_flights(destination="NYC", date="tomorrow")
  ↓
System: Returns [Flight A $200, Flight B $150, ...]
  ↓
LLM: Call book_flight(flight_id="B", seats=1)
  ↓
System: Returns confirmation_number="ABC123"
  ↓
LLM: "Flight booked! Confirmation: ABC123"
```

**OpenAI Function Calling:**
```python
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather?"}],
    functions=[
        {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                }
            }
        }
    ]
)
```

---

### Q11: How do you avoid hallucinations?

**Answer:**

**Hallucinations:** LLM generating plausible-sounding but false information.

**Prevention Strategies:**

1. **RAG (Retrieval-Augmented Generation):**
```python
# Provide ground truth context
prompt = f"""
Based on the following context:
{retrieved_documents}

Answer the question: {user_query}
"""
```

2. **Prompt Engineering:**
```
Good: "Answer based on the provided document only. If not found, say 'not found'."
Bad: "Answer the question"
```

3. **Temperature Control:**
```python
# Lower temperature = more deterministic, less creative
response = llm(prompt, temperature=0.3)  # More factual
response = llm(prompt, temperature=0.9)  # More creative
```

4. **Fact-Checking:**
```python
# Verify claims against knowledge base
generated_text = llm(prompt)
for claim in extract_claims(generated_text):
    if not verify_fact(claim, knowledge_base):
        flag_as_potential_hallucination(claim)
```

5. **Confidence Scoring:**
```
Ask LLM: "How confident are you (0-100)?"
Only trust if confidence > 80%
```

---

### Q12: Explain RAG pipeline in simple terms.

**Answer:**

**RAG (Retrieval-Augmented Generation):** Combine document retrieval with LLM generation.

**Pipeline:**
```
1. User Query: "What's company's refund policy?"
   ↓
2. Embed Query → Vector representation
   ↓
3. Retrieve: Search vector DB for relevant documents
   ↓ (Results: Refund policy doc)
   ↓
4. Prompt Construction:
   "Based on:
    [document text here]
    Answer: What's the refund policy?"
   ↓
5. LLM Generation → Accurate answer grounded in docs
   ↓
6. Return: "Refunds available within 30 days..."
```

**Benefits:**
- Accurate (grounded in documents)
- Current (uses latest docs)
- Traceable (can cite sources)
- Reduces hallucinations

---

### Q13: What are safety filters in LLMs?

**Answer:**

**Safety Filters:** Mechanisms to prevent harmful outputs.

**Types:**

1. **Input Filtering:**
```python
blocked_keywords = ["bomb", "weapon", ...]
if any(keyword in user_input for keyword in blocked_keywords):
    reject_request()
```

2. **Output Filtering:**
```python
# Check LLM response for safety violations
if contains_hateful_speech(response):
    return "I can't help with that"
```

3. **Fine-tuning + Alignment:**
- Train LLM with RLHF (Reinforcement Learning from Human Feedback)
- Learn to refuse harmful requests

4. **Rate Limiting:**
```python
# Prevent abuse
requests_per_minute = 60
if exceeds_limit:
    reject()
```

5. **Monitoring:**
```python
# Log and alert for suspicious patterns
log_request(user_id, query, response)
if suspicious_pattern_detected():
    notify_admin()
```

**Examples:**
- GPT-4 refuses to write malware
- Claude avoids generating illegal content
- LLaMA-2 trained with safety guidelines

---

### Q14: Difference between open-source LLM and proprietary LLM.

**Answer:**

| Aspect | Open-Source | Proprietary |
|--------|-------------|-----------|
| **Examples** | Llama 2, Mistral, Falcon | GPT-4, Claude, Gemini |
| **Cost** | Free (self-hosted) | $ per API call |
| **Control** | Full (can fine-tune) | Limited |
| **Performance** | Good, improving | State-of-the-art |
| **Latency** | Lower (local) | Higher (API call) |
| **Privacy** | Data stays private | May be stored |
| **Scale** | Self-manage | Provider manages |
| **Customization** | Full | Limited |

**When to use:**
- **Open-source:** Privacy-critical, cost-sensitive, need customization
- **Proprietary:** Best performance, no infrastructure, quick setup

---

### Q15: How you would productionize an LLM-based API?

**Answer:**

**Production Checklist:**

1. **Containerization:**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

2. **Orchestration (Kubernetes/AKS):**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: llm-api
        image: llm-api:v1
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
```

3. **API Framework (FastAPI):**
```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/summarize")
async def summarize(text: str):
    summary = llm.generate_summary(text)
    return {"summary": summary}
```

4. **Monitoring:**
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram

request_count = Counter('llm_requests_total', 'Total requests')
response_time = Histogram('llm_response_seconds', 'Response time')

@app.post("/summarize")
async def summarize(text: str):
    with response_time.time():
        result = llm.generate_summary(text)
    request_count.inc()
    return result
```

5. **Logging:**
```python
import logging

logger = logging.getLogger(__name__)

@app.post("/summarize")
async def summarize(text: str):
    logger.info(f"Summarizing: {len(text)} chars")
    try:
        result = llm.generate_summary(text)
        logger.info(f"Success: {len(result)} chars")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
```

6. **Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str):
    return embeddings.embed(text)
```

7. **Rate Limiting:**
```python
from slowapi import Limiter

limiter = Limiter(key_func=lambda: "global")

@app.post("/summarize")
@limiter.limit("100/minute")
async def summarize(text: str):
    return llm.generate_summary(text)
```

8. **Error Handling:**
```python
@app.post("/summarize")
async def summarize(text: str):
    try:
        if len(text) > 50000:
            raise ValueError("Text too long")
        result = llm.generate_summary(text)
        return {"status": "success", "result": result}
    except ValueError as e:
        return {"status": "error", "message": str(e)}, 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"status": "error", "message": "Internal error"}, 500
```