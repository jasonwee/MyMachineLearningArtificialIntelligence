https://codelabs.developers.google.com/tflite-nlp-model#0 

# setup
tflite-model-maker doesn't install in python3.10, python3.11 or python3.12 nor python3.13

the following is deprecated, just use the vscode devcontainer instead
```bash
/opt/weetech/python-3.9.21/bin/python3.9 -m venv tf_env_py39
source tf_env_py39/bin/activate
pip install pip==23.3.2
pip install tensorflow[and-cuda]
pip install numpy pandas
pip install tensorflowjs tflite_support
pip install tflite-model-maker
```

# train the model
```
$ python3 spam_model.py
```

# use the model
```
$ python3 spam_model.py
```
