import os
import pandas
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, concatenate
import numpy as np
import tensorflow as tf
from collections import Counter

# Disable interactive progress bars and deprecation noise
try:
    tf.keras.utils.disable_interactive_logging()
except Exception:
    pass

EPOCHS = int(os.getenv("EPOCHS_OVERRIDE", "50"))  # original 5000

######################################################################################
# Processing data
pwd = os.path.dirname(os.path.abspath(__file__))
file = pwd + "/data/kingcounty-house-data.csv"
g = pandas.read_csv(file, encoding="ISO-8859-1")
g["price"] = g["price"]/1000

X = g[["sqft_above","sqft_basement","sqft_lot","sqft_living","floors","bedrooms","yr_built","lat","long","bathrooms"]].values
Y = g["price"].values
zipcodes = pandas.get_dummies(g["zipcode"]).values
condition = pandas.get_dummies(g["condition"]).values
grade  = pandas.get_dummies(g["grade"]).values

X = np.concatenate((X, zipcodes), axis=1)
X = np.concatenate((X, condition), axis=1)
X = np.concatenate((X, grade), axis=1)

#######################################################################################
# Building deep network

#### Sequential API ####
def sequential_neural_model():
    model = Sequential()
    model.add(Input(shape=(97,)))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal', activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(5, kernel_initializer='normal', activation="relu"))
    model.add(Dense(1, kernel_initializer='normal'))

    sgd = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='mae', optimizer=sgd, metrics=["mae"])
    return model

    # model.fit(X, Y, epochs=200, verbose=2)

#### Model API ####
def model_neural_model():
    a = Input(shape=(10,))
    c = Dense(32, activation="tanh")(a)

    b = Input(shape=(87,))
    d = Dense(32, activation="tanh")(b)

    e = concatenate([a, b], axis=-1)
    finak = Dense(1)(e)

    model = Model(inputs=[a, b,], outputs=finak)

    model.compile(optimizer="rmsprop", loss="mae")
    return model

model = model_neural_model()
# Split features per functional inputs
X_head, X_tail = X[:, 0:10], X[:, 10:97]
history = model.fit([X_head, X_tail], Y, batch_size=32, epochs=EPOCHS, verbose=0, validation_split=0.2)

score = model.evaluate([X_head, X_tail], Y, verbose=0)
mae = score if isinstance(score, (float, int)) else score[0]

print("\n===== DATASET SUMMARY =====")
print(f"Samples: {X.shape[0]} | Features total: {X.shape[1]} (head=10, tail=87) | Epochs={EPOCHS}")

print("\n===== TRAINING SUMMARY =====")
print(f"Final MAE (train+val mixed eval): {mae:.4f}")
if 'loss' in history.history:
    print(f"First loss: {history.history['loss'][0]:.4f} | Last loss: {history.history['loss'][-1]:.4f}")
    if 'val_loss' in history.history:
        print(f"First val_loss: {history.history['val_loss'][0]:.4f} | Last val_loss: {history.history['val_loss'][-1]:.4f}")
if EPOCHS <= 3:
    print("(Nota: épocas muy bajas, sólo sanity check)")

print("\n===== MODEL SUMMARY =====")
model.summary()

#######################################################################################
# Evaluating the prediction

pred = model.predict([X_head, X_tail], verbose=0)
g["predicted"] = pred

print("\n===== SAMPLE PREDICTIONS =====")
limit = int(os.getenv("PRED_SAMPLE_LIMIT", "10"))
sample = g.head(limit)[["predicted","price"]]
sample["abs_err"] = (sample["predicted"] - sample["price"]).abs()
print(sample)

print("\nDone.")
