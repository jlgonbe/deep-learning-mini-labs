import os
import tensorflow as tf
import numpy as np
import pandas
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler

# Disable interactive progress bars globally
try:
	tf.keras.utils.disable_interactive_logging()
except Exception:
	pass

EPOCHS = int(os.getenv("EPOCHS_OVERRIDE", "50"))  # original 3000
PRED_SAMPLE_LIMIT = int(os.getenv("PRED_SAMPLE_LIMIT", "10"))

######################################################################################
# Processing data
pwd = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(pwd, "data", "vgsales.csv")
data = pandas.read_csv(data_file, encoding="ISO-8859-1")

# Filter rows with at least minimal sales to avoid extreme zeros
data = data.loc[data["NA_Sales"] > 1]
data = data.loc[data["EU_Sales"] > 1]

# Coerce numeric columns (handles strings like 'tbd') then drop resulting NaNs
numeric_cols = [
	"Critic_Score","Critic_Count","User_Score","User_Count",
	"NA_Sales","EU_Sales","JP_Sales"
]
for col in numeric_cols:
	data[col] = pandas.to_numeric(data[col], errors="coerce")
data = data.dropna(axis=0)

names = data[["Name", "Year_of_Release"]]
NA = data["NA_Sales"].astype("float32").values
EU = data["EU_Sales"].astype("float32").values
JP = data["JP_Sales"].astype("float32").values
X = data[["Critic_Score","Critic_Count","User_Score","User_Count"]].values

platform = pandas.get_dummies(data["Platform"]).values
genre = pandas.get_dummies(data["Genre"]).values
publisher = pandas.get_dummies(data["Publisher"]).values
rating = pandas.get_dummies(data["Rating"]).values

X = np.concatenate([X, platform, genre, publisher, rating], axis=1)
print("\n===== DATASET SUMMARY =====")
print(f"Samples: {X.shape[0]} | Features: {X.shape[1]} | Epochs={EPOCHS}")
print("Targets: NA/EU/JP sales (filtered NA>1, EU>1).")
print(f"Ranges -> NA:[{NA.min():.2f},{NA.max():.2f}] EU:[{EU.min():.2f},{EU.max():.2f}] JP:[{JP.min():.2f},{JP.max():.2f}]")
#######################################################################################
# Building deep network
def build_model(input_dim: int) -> Model:
	inp = Input(shape=(input_dim,))
	x = Dense(32, activation="sigmoid")(inp)
	x = Dense(32, activation="sigmoid")(x)
	out_na = Dense(1, name="na_out")(x)
	out_eu = Dense(1, name="eu_out")(x)
	out_jp = Dense(1, name="jp_out")(x)
	m = Model(inputs=inp, outputs=[out_na, out_eu, out_jp])
	m.compile(optimizer="adam", loss="mse")
	return m

model = build_model(X.shape[1])
history = model.fit(X, [NA, EU, JP], batch_size=100, epochs=EPOCHS, verbose=0, validation_split=0.2)
print("\n===== TRAINING SUMMARY =====")
print(f"Epochs run: {EPOCHS}")
if 'loss' in history.history:
	print(f"First loss: {history.history['loss'][0]:.4f} | Last loss: {history.history['loss'][-1]:.4f}")
val_keys = [k for k in history.history.keys() if k.startswith('val_')]
if val_keys:
	print("Validation losses (last): " + ", ".join([f"{k}={history.history[k][-1]:.4f}" for k in val_keys]))

#######################################################################################
# Evaluating the prediction
p_NA, p_EU, p_JP = model.predict(X, verbose=0)
predictions = pandas.DataFrame(np.concatenate((names, p_NA, NA[:, np.newaxis], p_EU, EU[:, np.newaxis], p_JP, JP[:, np.newaxis]), axis=1))
predictions.columns = ["Name", "Year", "p_NA", "NA", "p_EU", "EU", "p_JP", "JP"]
predictions["err_NA"] = (predictions["p_NA"] - predictions["NA"]).abs()
predictions["err_EU"] = (predictions["p_EU"] - predictions["EU"]).abs()
predictions["err_JP"] = (predictions["p_JP"] - predictions["JP"]).abs()
print("\n===== SAMPLE PREDICTIONS =====")
print(predictions.head(PRED_SAMPLE_LIMIT))
print("\nDone.")
