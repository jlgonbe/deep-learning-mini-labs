import os
import tensorflow as tf
import numpy as np
import pandas
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
import warnings

# Disable interactive progress bars globally
try:
    tf.keras.utils.disable_interactive_logging()
except Exception:
    pass

EPOCHS = int(os.getenv("EPOCHS_OVERRIDE", "5"))  # original nb_epoch=10
PRED_SAMPLE_LIMIT = int(os.getenv("PRED_SAMPLE_LIMIT", "10"))

######################################################################################
# Processing data
pwd = os.path.dirname(os.path.abspath(__file__))
file = pwd + "/data/kingcounty-house-data.csv"
g = pandas.read_csv(file, encoding="ISO-8859-1")
g["price"] = g["price"]/1000

X = g[["sqft_above", "sqft_basement", "sqft_lot", "sqft_living","floors", "bedrooms", "yr_built", "lat", "long", "bathrooms"]].values
Y = g["price"].values
zipcodes = pandas.get_dummies(g["zipcode"]).values
condition = pandas.get_dummies(g["condition"]).values
grade = pandas.get_dummies(g["grade"]).values

X = np.concatenate((X, zipcodes), axis=1)
X = np.concatenate((X, condition), axis=1)
X = np.concatenate((X, grade), axis=1)

#######################################################################################
# Building deep network


def neural_model1():
    model = Sequential()
    model.add(Input(shape=(97,)))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal', activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(5, kernel_initializer='normal', activation="relu"))
    model.add(Dense(1, kernel_initializer='normal'))
    sgd = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mae', optimizer=sgd, metrics=["mae"])
    return model


def neural_model2():
    model = Sequential()
    model.add(Input(shape=(97,)))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(14, kernel_initializer='normal', activation="sigmoid"))
    model.add(Dropout(0.3))
    model.add(Dense(5, kernel_initializer='normal', activation="relu"))
    model.add(Dense(1, kernel_initializer='normal'))
    sgd = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mae', optimizer=sgd, metrics=["mae"])
    return model


#######################################################################################
# Evauate models
print("\n===== DATASET SUMMARY =====")
print(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")

print("\n===== CROSS-VALIDATION SUMMARY =====")
print(f"Epochs per fold: {EPOCHS} | CV folds=5")

model1 = KerasRegressor(model=neural_model1, epochs=EPOCHS, batch_size=50, verbose=0)
cv1 = cross_val_score(model1, X, Y, cv=5, scoring="neg_mean_absolute_error")
print(f"Model1 mean_neg_mae={cv1.mean():.4f} std={cv1.std():.4f}")

model2 = KerasRegressor(model=neural_model2, epochs=EPOCHS, batch_size=50, verbose=0)
cv2 = cross_val_score(model2, X, Y, cv=5, scoring="neg_mean_absolute_error")
print(f"Model2 mean_neg_mae={cv2.mean():.4f} std={cv2.std():.4f}")

better = 'Model1' if cv1.mean() > cv2.mean() else 'Model2'
print(f"Best (higher neg_mae): {better}")

print("\n===== SAMPLE PREDICTIONS (best model) =====")
best_fn = neural_model1 if better == 'Model1' else neural_model2
best_model = best_fn()
best_model.fit(X, Y, epochs=EPOCHS, batch_size=50, verbose=0)
pred = best_model.predict(X, verbose=0)
pred = np.array(pred).ravel()  # flatten to 1D
df_pred = pandas.DataFrame({"pred": pred[:PRED_SAMPLE_LIMIT], "real": Y[:PRED_SAMPLE_LIMIT]})
df_pred["abs_err"] = (df_pred["pred"] - df_pred["real"]).abs()
print(df_pred)
print("\nDone.")
