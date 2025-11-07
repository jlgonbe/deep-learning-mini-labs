import math
import os
import numpy as np
import pandas
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Disable interactive progress bars globally
try:
	tf.keras.utils.disable_interactive_logging()
except Exception:
	pass

EPOCHS = int(os.getenv("EPOCHS_OVERRIDE", "20"))  # original 100
PRED_SAMPLE_LIMIT = int(os.getenv("PRED_SAMPLE_LIMIT", "10"))

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

pwd = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(pwd, "data", "new_houses_london.csv")
dataframe = pandas.read_csv(data_file)["New Dwellings London"]
dataset = dataframe.values.astype('float32')[:, np.newaxis]

scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)
train_size = int(len(dataset_scaled) * 0.67)
train, test = dataset_scaled[0:train_size, :], dataset_scaled[train_size:, :]

look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print("\n===== DATASET SUMMARY =====")
print(f"Total samples: {dataset.shape[0]} | Train: {train.shape[0]} | Test: {test.shape[0]} | Look_back: {look_back}")
print(f"TrainX shape: {trainX.shape} | TestX shape: {testX.shape}")

model = Sequential([
    Input(shape=(1, look_back)),
    LSTM(4, return_sequences=True),
    LSTM(4),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(trainX, trainY, epochs=EPOCHS, batch_size=1, verbose=0, validation_split=0.2)

print("\n===== TRAINING SUMMARY =====")
if 'loss' in history.history:
    print(f"Loss first/last: {history.history['loss'][0]:.4f}/{history.history['loss'][-1]:.4f}")
    if 'val_loss' in history.history:
        print(f"Val loss first/last: {history.history['val_loss'][0]:.4f}/{history.history['val_loss'][-1]:.4f}")
if EPOCHS <= 3:
    print("(Nota: pocas épocas, sanity check)")

trainPredict = model.predict(trainX, verbose=0)
testPredict = model.predict(testX, verbose=0)
trainPredict_inv = scaler.inverse_transform(trainPredict)
trainY_inv = scaler.inverse_transform([trainY])
testPredict_inv = scaler.inverse_transform(testPredict)
testY_inv = scaler.inverse_transform([testY])
trainScore = math.sqrt(mean_squared_error(trainY_inv[0], trainPredict_inv[:, 0]))
testScore = math.sqrt(mean_squared_error(testY_inv[0], testPredict_inv[:, 0]))

print("\n===== EVALUATION SUMMARY =====")
print(f"Train RMSE: {trainScore:.2f}")
print(f"Test RMSE: {testScore:.2f}")

print("\n===== SAMPLE PREDICTIONS =====")
limit = min(PRED_SAMPLE_LIMIT, testPredict_inv.shape[0])
sample = pandas.DataFrame({
    "Pred": testPredict_inv[:limit, 0],
    "Real": testY_inv[0][:limit]
})
sample["AbsErr"] = (sample["Pred"] - sample["Real"]).abs()
print(sample)

np.savetxt('dataset_orig.csv', scaler.inverse_transform(dataset_scaled), delimiter=',')
np.savetxt('trainpred.csv', trainPredict_inv, delimiter=',')
np.savetxt('testpred.csv', testPredict_inv, delimiter=',')
print("\nDone.")
