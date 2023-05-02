print("Model-1")
print("importing libraries")

from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

seed = 0
np.random.seed(seed)
import tensorflow as tf

tf.random.set_seed(seed)
import os

print("Changing path")
os.environ["PATH"] = "./Vivado/2019.2/bin:" + os.environ["PATH"]
print("New Path: ", os.environ["PATH"])

print("Loading the data")


data = fetch_openml("hls4ml_lhc_jets_hlf")
X, y = data["data"], data["target"]


print(data["feature_names"])
print(X.shape, y.shape)
print(X[:5])
print(y[:5])


le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y, 5)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(y[:5])

scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)

np.save("../home/jovyan/X_train_val.npy", X_train_val)
np.save("../home/jovyan/X_test.npy", X_test)
np.save("../home/jovyan/y_train_val.npy", y_train_val)
np.save("../home/jovyan/y_test.npy", y_test)
classes = np.load("classes.npy", allow_pickle=True)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from callbacks import all_callbacks

print("Intialising the model")

model = Sequential()
model.add(
    Dense(
        64,
        input_shape=(16,),
        name="fc1",
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
    )
)
model.add(Activation(activation="relu", name="relu1"))
model.add(
    Dense(
        32,
        name="fc2",
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
    )
)
model.add(Activation(activation="relu", name="relu2"))
model.add(
    Dense(
        32,
        name="fc3",
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
    )
)
model.add(Activation(activation="relu", name="relu3"))
model.add(
    Dense(
        5,
        name="output",
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
    )
)
model.add(Activation(activation="softmax", name="softmax"))


print("training the model")

train = True
if train:
    adam = Adam(lr=0.0001)
    model.compile(
        optimizer=adam, loss=["categorical_crossentropy"], metrics=["accuracy"]
    )
    callbacks = all_callbacks(
        stop_patience=1000,
        lr_factor=0.5,
        lr_patience=10,
        lr_epsilon=0.000001,
        lr_cooldown=2,
        lr_minimum=0.0000001,
        outputDir="../home/jovyan/model_1",
    )
    model.fit(
        X_train_val,
        y_train_val,
        batch_size=1024,
        epochs=20,
        validation_split=0.25,
        shuffle=True,
        callbacks=callbacks.callbacks,
    )
else:
    from tensorflow.keras.models import load_model

    model = load_model("model_1/KERAS_check_best_model.h5")


print("Checking the performance")

import plotting
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

y_keras = model.predict(X_test)
print(
    "Accuracy: {}".format(
        accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))
    )
)


print("making hls4ml config and model")

import hls4ml

config = hls4ml.utils.config_from_keras_model(model, granularity="model")
print("-----------------------------------")
print("Configuration")
plotting.print_dict(config)
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir="../home/jovyan/model_1/hls4ml_prj",
    part="xcu250-figd2104-2L-e",
)

print("plotting the hls4ml model")

hls4ml.utils.plot_model(
    hls_model,
    show_shapes=True,
    show_precision=True,
    to_file="../home/jovyan/model_1/model_1_hls.png",
)


print("Compiling and predicting the hls4ml model")

hls_model.compile()
X_test = np.ascontiguousarray(X_test)
y_hls = hls_model.predict(X_test)


print("Comparing both the models ")

print(
    "Keras  Accuracy: {}".format(
        accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))
    )
)
print(
    "hls4ml Accuracy: {}".format(
        accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))
    )
)

print("Saving the predictions...")
np.save("../home/jovyan/model_1/hls4ml_prj/y_keras.npy", y_keras)
np.save("../home/jovyan/model_1/hls4ml_prj/y_hls.npy", y_hls)


print("Synthesing...")

hls_model.build(csim=False)


print("Reading reports")

hls4ml.report.read_vivado_report("../home/jovyan/model_1/hls4ml_prj/")
