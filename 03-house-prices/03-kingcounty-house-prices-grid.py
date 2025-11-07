import os
import tensorflow as tf
import warnings
import numpy as np
import pandas
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from scikeras.wrappers import KerasRegressor
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error

# Disable interactive progress bars globally
try:
	tf.keras.utils.disable_interactive_logging()
except Exception:
	pass

EPOCHS = int(os.getenv("EPOCHS_OVERRIDE", "5"))  # original nb_epoch=20
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
def neural_model1(init, first_layer_N):
    model = Sequential()
    model.add(Input(shape=(97,)))
    model.add(Dense(first_layer_N, kernel_initializer=init, activation='relu'))
    model.add(Dense(5, activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(1))
    sgd = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mae', optimizer=sgd, metrics=["mae"])
    return model

# model.fit(X, Y, nb_epoch=300, verbose=2)

#######################################################################################
"""Grid search over simple regression model.
Previous FAIL: scikeras raised 'Invalid parameter first_layer_N' because the wrapper
didn't declare that parameter (and 'init') in its constructor. We expose them now
with default values so GridSearchCV can vary them.
"""
try:
    tf.keras.utils.disable_interactive_logging()
except Exception:
    pass
warnings.filterwarnings("ignore", message="``build_fn`` will be renamed", category=UserWarning)

model = KerasRegressor(
    model=neural_model1,
    epochs=EPOCHS,
    verbose=0,
    first_layer_N=10,
    init="uniform",
)

parameters = {
    'batch_size': [10, 25],
    'init': ['uniform', 'normal'],
    'first_layer_N': [10, 50, 64],
}
grid = model_selection.GridSearchCV(model, parameters, n_jobs=-1, scoring="neg_mean_absolute_error")
grid_result = grid.fit(X, Y)

print("\n===== DATASET SUMMARY =====")
print(f"Samples: {X.shape[0]} | Features: {X.shape[1]} | Epochs(grid each)={EPOCHS}")

print("\n===== GRID SEARCH SUMMARY =====")
print("Best (neg_mae): %.4f using %s" % (grid_result.best_score_, grid_result.best_params_))
cv = grid_result.cv_results_
means = cv['mean_test_score']
stds = cv['std_test_score']
params_list = cv['params']
for i, params in enumerate(params_list):
    print(f"Combo {i}: mean={means[i]:.4f} std={stds[i]:.4f} params={params}")

print("\n===== SAMPLE PREDICTIONS (best model) =====")
# Use the fitted scikeras estimator directly; its .predict returns a 1D numpy array.
best_estimator = grid_result.best_estimator_
pred = best_estimator.predict(X)
pred = np.array(pred).ravel()  # ensure 1D for DataFrame construction
df_pred = pandas.DataFrame({"pred": pred[:PRED_SAMPLE_LIMIT], "real": Y[:PRED_SAMPLE_LIMIT]})
df_pred["abs_err"] = (df_pred["pred"] - df_pred["real"]).abs()
print(df_pred)

print("\nDone.")
