import os
from collections import Counter
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split

# Disable interactive progress bars globally
try:
	tf.keras.utils.disable_interactive_logging()
except Exception:
	pass

EPOCHS = int(os.getenv("EPOCHS_OVERRIDE", "10"))  # original 10/50
PRED_SAMPLE_LIMIT = int(os.getenv("PRED_SAMPLE_LIMIT", "10"))

#----------------------------------------------------------------------------------------#

pwd = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(pwd, "data", "mushrooms.csv")
d = pd.read_csv(file)

# Target: 'e' edible vs 'p' poisonous
raw_target = d["class"].values
class_counts = Counter(raw_target)
target_labels = sorted(class_counts.keys())  # typically ['e','p']
label_meaning = {"e": "edible", "p": "poisonous"}
mushroom_class = pd.get_dummies(raw_target).values[:,0]  # binary 0/1
del d["class"]

# One-hot encode remaining categorical columns
Q = None
for col_name in d.columns:
    encoded = pd.get_dummies(d[col_name]).values
    Q = encoded if Q is None else np.concatenate((Q, encoded), axis=1)

X_train, X_test, y_train, y_test = train_test_split(Q, mushroom_class, test_size=0.332, random_state=42)

#----------------------------------------------------------------------------------------#

num_samples, num_features = X_train.shape
print(f"Feature matrix shape (train): {X_train.shape}")

model = Sequential([
    Input(shape=(num_features,)),
    Dense(32, kernel_initializer='uniform', activation='relu'),
    Dropout(0.1),
    Dense(15, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])  

# model.fit(X_train, y_train, epochs=50, verbose=2)
history = model.fit(X_train, y_train, epochs=EPOCHS, verbose=0, validation_data=(X_test, y_test))
results = model.evaluate(X_test, y_test, verbose=0)

print("\n===== DATASET SUMMARY =====")
total = len(raw_target)
print(f"Samples: {total} | Train: {len(X_train)} | Test: {len(X_test)} | Features(expanded): {num_features} | Epochs={EPOCHS}")
print("Class distribution:")
for cls in sorted(class_counts.keys()):
    print(f"  {cls} ({label_meaning.get(cls, cls)}): {class_counts[cls]}")

print("\n===== MODEL SUMMARY =====")
model.summary()

print("\n===== TRAINING SUMMARY =====")
final_loss = history.history['loss'][0], history.history['loss'][-1]
final_acc = history.history['accuracy'][0], history.history['accuracy'][-1]
final_val_loss = history.history['val_loss'][0], history.history['val_loss'][-1]
final_val_acc = history.history['val_accuracy'][0], history.history['val_accuracy'][-1]
print(f"Loss first/last: {final_loss[0]:.4f}/{final_loss[1]:.4f} | Acc first/last: {final_acc[0]:.4f}/{final_acc[1]:.4f}")
print(f"Val loss first/last: {final_val_loss[0]:.4f}/{final_val_loss[1]:.4f} | Val acc first/last: {final_val_acc[0]:.4f}/{final_val_acc[1]:.4f}")
print(f"Final test loss: {results[0]:.4f} | Final test acc: {results[1]:.4f}")
if EPOCHS <= 3:
    print("(Nota: pocas épocas, verificación rápida)")

print("\n===== SAMPLE PREDICTIONS =====")
pred_probs = model.predict(X_test[:PRED_SAMPLE_LIMIT], verbose=0)
for i, p in enumerate(pred_probs):
    label = 1 if p[0] >= 0.5 else 0
    human = label_meaning[target_labels[label]]
    print(f"Row {i}: prob_edible={p[0]:.2%} => predicted {human}")

print("\nDone.")
