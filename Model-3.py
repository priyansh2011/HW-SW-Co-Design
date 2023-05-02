print("Model-3")

from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline
seed = 0
np.random.seed(seed)
import tensorflow as tf

tf.random.set_seed(seed)
import os

os.environ["PATH"] = "./Vivado/2019.2/bin:" + os.environ["PATH"]

X_train_val = np.load("../home/jovyan/X_train_val.npy")
X_test = np.load("../home/jovyan/X_test.npy")
y_train_val = np.load("../home/jovyan/y_train_val.npy")
y_test = np.load("../home/jovyan/y_test.npy")
classes = np.load("classes.npy", allow_pickle=True)


from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from callbacks import all_callbacks
from tensorflow.keras.layers import Activation
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu


model = Sequential()
model.add(
    QDense(
        64,
        input_shape=(16,),
        name="fc1",
        kernel_quantizer=quantized_bits(6, 0, alpha=1),
        bias_quantizer=quantized_bits(6, 0, alpha=1),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
    )
)
model.add(QActivation(activation=quantized_relu(6), name="relu1"))
model.add(
    QDense(
        32,
        name="fc2",
        kernel_quantizer=quantized_bits(6, 0, alpha=1),
        bias_quantizer=quantized_bits(6, 0, alpha=1),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
    )
)
model.add(QActivation(activation=quantized_relu(6), name="relu2"))
model.add(
    QDense(
        32,
        name="fc3",
        kernel_quantizer=quantized_bits(6, 0, alpha=1),
        bias_quantizer=quantized_bits(6, 0, alpha=1),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
    )
)
model.add(QActivation(activation=quantized_relu(6), name="relu3"))
model.add(
    QDense(
        5,
        name="output",
        kernel_quantizer=quantized_bits(6, 0, alpha=1),
        bias_quantizer=quantized_bits(6, 0, alpha=1),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
    )
)
model.add(Activation(activation="softmax", name="softmax"))

""" Train sparse
"""

from tensorflow_model_optimization.python.core.sparsity.keras import (
    prune,
    pruning_callbacks,
    pruning_schedule,
)
from tensorflow_model_optimization.sparsity.keras import strip_pruning

pruning_params = {
    "pruning_schedule": pruning_schedule.ConstantSparsity(
        0.75, begin_step=2000, frequency=100
    )
}
model = prune.prune_low_magnitude(model, **pruning_params)


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
        outputDir="../home/jovyan/model_3",
    )
    callbacks.callbacks.append(pruning_callbacks.UpdatePruningStep())
    model.fit(
        X_train_val,
        y_train_val,
        batch_size=1024,
        epochs=20,
        validation_split=0.25,
        shuffle=True,
        callbacks=callbacks.callbacks,
    )
    # Save the model again but with the pruning 'stripped' to use the regular layer types
    model = strip_pruning(model)
    model.save("../home/jovyan/model_3/KERAS_check_best_model.h5")
else:
    from tensorflow.keras.models import load_model
    from qkeras.utils import _add_supported_quantized_objects

    co = {}
    _add_supported_quantized_objects(co)
    model = load_model("model_3/KERAS_check_best_model.h5", custom_objects=co)


import hls4ml
import plotting

hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ["Activation"]
hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = "AP_RND"
hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = "AP_SAT"

config = hls4ml.utils.config_from_keras_model(model, granularity="name")
config["LayerName"]["softmax"]["exp_table_t"] = "ap_fixed<18,8>"
config["LayerName"]["softmax"]["inv_table_t"] = "ap_fixed<18,4>"
print("-----------------------------------")
plotting.print_dict(config)
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir="../home/jovyan/model_3/hls4ml_prj",
    part="xcu250-figd2104-2L-e",
)
hls_model.compile()

y_qkeras = model.predict(np.ascontiguousarray(X_test))
y_hls = hls_model.predict(np.ascontiguousarray(X_test))

np.save("../home/jovyan/model_3/y_qkeras.npy", y_qkeras)
np.save("../home/jovyan/model_3/y_hls.npy", y_hls)

# %matplotlib inline
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

model_ref = load_model("../home/jovyan/model_1/KERAS_check_best_model.h5")
y_ref = model_ref.predict(X_test)

print(
    "Accuracy baseline:  {}".format(
        accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_ref, axis=1))
    )
)
print(
    "Accuracy pruned, quantized: {}".format(
        accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_qkeras, axis=1))
    )
)
print(
    "Accuracy hls4ml: {}".format(
        accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))
    )
)
np.save("../home/jovyan/model_3/y_keras.npy", y_ref)


hls_model.build(csim=False)


hls4ml.report.read_vivado_report("../home/jovyan/model_3/hls4ml_prj")


hls4ml.report.read_vivado_report("../home/jovyan/model_1/hls4ml_prj")


hls4ml.report.read_vivado_report("../home/jovyan/model_2/hls4ml_prj")
