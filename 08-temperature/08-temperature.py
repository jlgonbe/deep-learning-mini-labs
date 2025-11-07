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

EPOCHS_SINGLE = int(os.getenv("EPOCHS_OVERRIDE", "20"))  # original 100 inside compute_rmse
PRED_SAMPLE_LIMIT = int(os.getenv("PRED_SAMPLE_LIMIT", "10"))

#--------------------------------------------------------------------------------------#

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

#--------------------------------------------------------------------------------------#

pwd = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(pwd, "data", "temperature.csv")
dataframe = pandas.read_csv(data_file)["Mean"]
dataset = dataframe.values.astype('float32')[:, np.newaxis]

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("\n===== DATASET SUMMARY =====")
print(f"Total samples: {dataset.shape[0]} | Train: {train.shape[0]} | Test: {test.shape[0]}")

def compute_rmse(lags):
    look_back = lags
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(Input(shape=(1, look_back)))
    model.add(LSTM(4, return_sequences=True))
    model.add(LSTM(4))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=EPOCHS_SINGLE, batch_size=50, verbose=0, validation_split=0.2)

    trainPredict = model.predict(trainX, verbose=0)
    testPredict = model.predict(testX, verbose=0)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print("\n===== TRAINING SUMMARY =====")
    if 'loss' in history.history:
        print(f"Loss first/last: {history.history['loss'][0]:.4f}/{history.history['loss'][-1]:.4f}")
        if 'val_loss' in history.history:
            print(f"Val loss first/last: {history.history['val_loss'][0]:.4f}/{history.history['val_loss'][-1]:.4f}")
    print("\n===== EVALUATION SUMMARY =====")
    print('Train RMSE: %.2f' % (trainScore))
    print('Test RMSE: %.2f' % (testScore))
    print("\n===== SAMPLE PREDICTIONS =====")
    limit = min(PRED_SAMPLE_LIMIT, testPredict.shape[0])
    sample = pandas.DataFrame({
        "Pred": testPredict[:limit, 0],
        "Real": testY[0][:limit]
    })
    sample["AbsErr"] = (sample["Pred"] - sample["Real"]).abs()
    print(sample)

    np.savetxt('dataset_orig_T.csv', scaler.inverse_transform(dataset), delimiter=',')
    np.savetxt('trainpred_T.csv', trainPredict, delimiter=',')
    np.savetxt('testpred_T.csv', testPredict, delimiter=',')


if __name__ == "__main__":
    # compute_rmse(10)
    compute_rmse(3)
    print("\nDone.")
