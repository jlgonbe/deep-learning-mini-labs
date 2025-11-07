import os
import numpy as np
import pandas
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import Matrix_CV_ML3D as DImage

# Disable interactive progress bars globally
try:
	tf.keras.utils.disable_interactive_logging()
except Exception:
	pass

EPOCHS = int(os.getenv("EPOCHS_OVERRIDE", "5"))  # original nb_epoch=50
PRED_SAMPLE_LIMIT = int(os.getenv("PRED_SAMPLE_LIMIT", "10"))

pwd = os.path.dirname(os.path.abspath(__file__))

train_data_img_folder = pwd + "/img/train"
train_data = DImage.Matrix_CV_ML3D(train_data_img_folder, 65, 50)
train_data.build_ML_matrix()

###################################################################################################
y = to_categorical(train_data.labels)
train_data = train_data.global_matrix
# Original data is channels_first (N, C, H, W); TensorFlow CPU backprop error -> convert to channels_last.
train_data = np.transpose(train_data, (0, 2, 3, 1))  # (N, H, W, C)
train_data = train_data.astype('float32')/255

def build_model() -> Sequential:
	m = Sequential([
		Input(shape=(50, 65, 3)),
		Conv2D(32, (3, 3), activation='relu'),
		Conv2D(32, (3, 3), activation='relu'),
		MaxPooling2D(pool_size=(2, 2)),
		Dropout(0.25),
		Flatten(),
		Dense(128, activation='relu'),
		Dropout(0.5),
		Dense(2, activation='softmax')
	])
	m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return m

model = build_model()
history = model.fit(train_data, y, epochs=EPOCHS, verbose=0, batch_size=32, validation_split=0.2)
print("\n===== DATASET SUMMARY =====")
print(f"Train samples: {train_data.shape[0]} | Image shape: (50,65,3)")
unique, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
print("Class distribution (train): " + ", ".join([f"{u}={c}" for u,c in zip(unique, counts)]))
print("\n===== MODEL SUMMARY =====")
model.summary()
print("\n===== TRAINING SUMMARY =====")
score_train = model.evaluate(train_data, y, verbose=0)
print(f"Train loss: {score_train[0]:.4f} | Train acc: {score_train[1]:.4f}")
if 'loss' in history.history:
	print(f"Loss first/last: {history.history['loss'][0]:.4f}/{history.history['loss'][-1]:.4f}")
	if 'val_loss' in history.history:
		print(f"Val loss first/last: {history.history['val_loss'][0]:.4f}/{history.history['val_loss'][-1]:.4f}")
if EPOCHS <= 3:
	print("(Nota: pocas épocas, sanity check)")
##################################################################################################
test_data_img_folder = pwd + "/img/test"
test_data = DImage.Matrix_CV_ML3D(test_data_img_folder, 65, 50)
test_data.build_ML_matrix()

labels = test_data.labels
y = to_categorical(test_data.labels)
test_data = test_data.global_matrix
test_data = np.transpose(test_data, (0, 2, 3, 1))
test_data = test_data.astype('float32')/255

score_test = model.evaluate(test_data, y, verbose=0)
print("\n===== TEST SUMMARY =====")
print(f"Test loss: {score_test[0]:.4f} | Test acc: {score_test[1]:.4f}")
predictions = model.predict(test_data, verbose=0)
predicted = np.argmax(predictions, axis=1)
print("\n===== SAMPLE PREDICTIONS =====")
limit = min(PRED_SAMPLE_LIMIT, predicted.shape[0])
sample_df = pandas.DataFrame({
	"Label": labels[:limit],
	"Pred": predicted[:limit],
	"Prob_class0": predictions[:limit, 0],
	"Prob_class1": predictions[:limit, 1]
})
sample_df["Correct"] = (sample_df["Label"] == sample_df["Pred"]).astype(int)
print(sample_df)
print("\nDone.")
