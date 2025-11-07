import os
import tensorflow as tf
import numpy as np
import pandas
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
import Matrix_CV_ML as cv

# Disable interactive progress bars globally
try:
	tf.keras.utils.disable_interactive_logging()
except Exception:
	pass

EPOCHS = int(os.getenv("EPOCHS_OVERRIDE", "20"))  # original 1000
PRED_SAMPLE_LIMIT = int(os.getenv("PRED_SAMPLE_LIMIT", "10"))

pwd = os.path.dirname(os.path.abspath(__file__))
train_data_img_folder = pwd + "/img/train"

train_data = cv.Matrix_CV_ML(train_data_img_folder, 100, 133)
train_data.build_ML_matrix()

test_data_img_folder = pwd + "/img/test"

test_data = cv.Matrix_CV_ML(test_data_img_folder, 100, 133)
test_data.build_ML_matrix()

data_dim = 100 * 133

def build_model(input_dim: int) -> Sequential:
	m = Sequential([
		Input(shape=(input_dim,)),
		Dense(64, kernel_initializer='uniform', activation='relu'),
		Dropout(0.5),
		Dense(64, activation='relu'),
		Dropout(0.5),
		Dense(1, activation='sigmoid')
	])
	m.compile(loss='binary_crossentropy', optimizer='sgd', metrics=["accuracy"])
	return m

model = build_model(data_dim)
history = model.fit(train_data.global_matrix, train_data.labels, epochs=EPOCHS, verbose=0, validation_split=0.2)
results = model.evaluate(train_data.global_matrix, train_data.labels, verbose=0)
print("\n===== DATASET SUMMARY =====")
print(f"Train samples: {train_data.global_matrix.shape[0]} | Test samples: {test_data.global_matrix.shape[0]} | Feature dim: {data_dim}")
unique, counts = np.unique(train_data.labels, return_counts=True)
print("Class distribution (train): " + ", ".join([f"{int(u)}={c}" for u,c in zip(unique, counts)]))
print("\n===== MODEL SUMMARY =====")
model.summary()
print("\n===== TRAINING SUMMARY =====")
print(f"Final loss: {results[0]:.4f} | Final acc: {results[1]:.4f}")
if 'loss' in history.history:
	print(f"Loss first/last: {history.history['loss'][0]:.4f}/{history.history['loss'][-1]:.4f}")
	if 'val_loss' in history.history:
		print(f"Val loss first/last: {history.history['val_loss'][0]:.4f}/{history.history['val_loss'][-1]:.4f}")
if EPOCHS <= 3:
	print("(Nota: pocas épocas, sanity check)")
print("\n===== SAMPLE PREDICTIONS =====")
raw_probs = model.predict(test_data.global_matrix, verbose=0).flatten()
predictions = (raw_probs > 0.5).astype(int)
sample_idx = np.arange(min(PRED_SAMPLE_LIMIT, predictions.shape[0]))
sample_df = pandas.DataFrame({
	"Observed": test_data.labels[sample_idx],
	"Prob": raw_probs[sample_idx],
	"Predicted": predictions[sample_idx]
})
sample_df["Correct"] = (sample_df["Observed"] == sample_df["Predicted"]).astype(int)
print(sample_df)
print("\nDone.")
