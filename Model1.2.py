print("Model-1.2")

from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import plotting
import os

os.environ["PATH"] = "./Vivado/2019.2/bin:" + os.environ["PATH"]


X_train_val = np.load("../home/jovyan/X_train_val.npy")
X_test = np.ascontiguousarray(np.load("../home/jovyan/X_test.npy"))
y_train_val = np.load("../home/jovyan/y_train_val.npy")
y_test = np.load("../home/jovyan/y_test.npy")
classes = np.load("classes.npy", allow_pickle=True)


print("loading the previous model unoptimised...")

from tensorflow.keras.models import load_model

model = load_model("../home/jovyan/model_1/KERAS_check_best_model.h5")
y_keras = model.predict(X_test)

"""
This time, we'll create a config with finer granularity. When we print the config dictionary, you'll notice that an entry is created for each named Layer of the model. See for the first layer, for example:
```LayerName:
    fc1:
        Precision:
            weight: ap_fixed<16,6>
            bias:   ap_fixed<16,6>
            result: ap_fixed<16,6>
        ReuseFactor: 1
```
Taken 'out of the box' this config will set all the parameters to the same settings as in part 1, but we can use it as a template to start modifying things. 
"""

import hls4ml

config = hls4ml.utils.config_from_keras_model(model, granularity="name")
print("-----------------------------------")
plotting.print_dict(config)
print("-----------------------------------")

print("Profiling unoptimised model..")

for layer in config["LayerName"].keys():
    config["LayerName"][layer]["Trace"] = True
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir="../home/jovyan/model_1/hls4ml_prj_2",
    part="xcu250-figd2104-2L-e",
)
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=X_test[:1000])

print("Saving keras outputs...")
np.save("../home/jovyan/model_1/hls4ml_prj_2/y_keras.npy", y_keras)


config["LayerName"]["fc1"]["Precision"]["weight"] = "ap_fixed<8,2>"
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir="../home/jovyan/model_1/hls4ml_prj_2",
    part="xcu250-figd2104-2L-e",
)
hls4ml.model.profiling.numerical(model=model, hls_model=hls_model)
# hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file="../home/jovyan/model_1/hls4ml_prj_2.png")


hls_model.compile()
y_hls = hls_model.predict(X_test)

print("Saving hls predictions...")
np.save("../home/jovyan/model_1/hls4ml_prj_2/y_hls_8.npy", y_hls)


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


config = hls4ml.utils.config_from_keras_model(model, granularity="Model")
print("-----------------------------------")
print(config)
print("-----------------------------------")
# Set the ReuseFactor to 2 throughout
config["Model"]["ReuseFactor"] = 2
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir="../home/jovyan/model_1/hls4ml_prj_2",
    part="xcu250-figd2104-2L-e",
)
hls_model.compile()
y_hls = hls_model.predict(X_test)
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

print("Saving results after reuse factor = 2")
np.save("../home/jovyan/model_1/hls4ml_prj_2/y_hls_resuse_2.npy", y_hls)


print("Synthesing the model..")

hls_model.build(csim=False)


hls4ml.report.read_vivado_report("../home/jovyan/model_1/hls4ml_prj_2")

hls4ml.report.read_vivado_report("../home/jovyan/model_1/hls4ml_prj")
