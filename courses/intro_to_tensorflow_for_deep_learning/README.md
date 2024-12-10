# setup virtual environment
At doing this coursework, I only have python3.11 and python3.13 installed. Because at this point of time, TF does not support py3.13, i just create venv using py3.11
```
$ python3 -m venv tf_env_py311
$ source tf_env_py311/bin/activate
$ pip install --upgrade pip
$ pip install tensorflow[and-cuda]
```

using this link as a reference https://www.tensorflow.org/install/pip

# verify tf works
tested work using py3.11
```
$ python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
tf.Tensor(-1000.3271, shape=(), dtype=float32)
$ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
package installed as of this time
```
$ pip list | grep -i tensor
tensorboard                  2.18.0
tensorboard-data-server      0.7.2
tensorflow                   2.18.0
tensorflow-io-gcs-filesystem 0.37.1
$ pip list | grep -i cuda
nvidia-cuda-cupti-cu12       12.5.82
nvidia-cuda-nvcc-cu12        12.5.82
nvidia-cuda-nvrtc-cu12       12.5.82
nvidia-cuda-runtime-cu12     12.5.82
```


