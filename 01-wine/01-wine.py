import os
from collections import Counter
import pandas
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Input

# Disable interactive progress bars globally
try:
	tf.keras.utils.disable_interactive_logging()
except Exception:
	pass

EPOCHS = int(os.getenv("EPOCHS_OVERRIDE", "1000"))
PRED_SAMPLE_LIMIT = int(os.getenv("PRED_SAMPLE_LIMIT", "10"))

##############################################################################################################################################

pwd = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(pwd, "data", "wine.data")
d = pandas.read_csv(file, names=["Class","Alcohol","Malic Alic","Ash","Alcanility of Ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280_OD315_diluted wines","Proline"])

# Preserve raw class labels (1..3) for human-readable summary
raw_classes = d["Class"].values
class_counts = Counter(raw_classes)
class_names = ["Class 1", "Class 2", "Class 3"]

# One-hot encode classes shifting indices (keras expects 0-based). Original labels are 1,2,3.
y = to_categorical(raw_classes)
y = y[:,1:4]  # Keep columns for classes 1..3
del d["Class"]

X = d.values

print("One-hot encoded labels sample (first 5):")
print(y[:5])
print()

##############################################################################################################################################

model = Sequential([
	Input(shape=(13,)),
	Dense(40, kernel_initializer='normal', activation='relu'),
	Dense(10, kernel_initializer='normal', activation='sigmoid'),
	Dropout(0.10),
	Dense(5, kernel_initializer='normal', activation='relu'),
	Dense(3, kernel_initializer='normal', activation='softmax')
])

sgd = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["acc"])
 
history = model.fit(X, y, epochs=EPOCHS, verbose=0, validation_split=0.2)
score = model.evaluate(X, y, verbose=0)

# ---------------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------------
num_samples, num_features = X.shape
print("\n===== DATASET SUMMARY =====")
print(f"Samples: {num_samples} | Features: {num_features} | Epochs={EPOCHS}")
print("Class distribution:")
for cls in sorted(class_counts.keys()):
	print(f"  {class_names[cls-1]} (label {cls}): {class_counts[cls]}")

print("\n===== MODEL SUMMARY =====")
model.summary()

print("\n===== TRAINING SUMMARY =====")
print(f"Final loss: {score[0]:.4f} | Final accuracy: {score[1]:.4f}")
if 'loss' in history.history:
	print(f"Loss first/last: {history.history['loss'][0]:.4f}/{history.history['loss'][-1]:.4f}")
	if 'val_loss' in history.history:
		print(f"Val loss first/last: {history.history['val_loss'][0]:.4f}/{history.history['val_loss'][-1]:.4f}")
if EPOCHS <= 5:
	print("(Nota: pocas épocas -> sanity check rápido)")

# Load wine1 samples for prediction demonstration
wine1_file = os.path.join(pwd, "data", "wine1.csv")
wine1 = pandas.read_csv(wine1_file, names=["Alcohol","Malic Alic","Ash","Alcanility of Ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280_OD315_diluted wines","Proline"])
wine1_np = wine1.values
predicted = model.predict(wine1_np, verbose=0)

print("\n===== SAMPLE PREDICTIONS (wine1.csv) =====")
for idx, probs in enumerate(predicted[:PRED_SAMPLE_LIMIT]):
	top_idx = int(np.argmax(probs))
	top_prob = probs[top_idx]
	probs_str = ", ".join([f"{class_names[i]}={p:.2%}" for i, p in enumerate(probs)])
	print(f"Row {idx}: Predicted {class_names[top_idx]} (p={top_prob:.2%}) | {probs_str}")

print("\nDone.")
