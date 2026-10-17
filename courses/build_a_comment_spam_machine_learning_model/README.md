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


# host setup
if using devcontainer (and recomended to do so), please ensure the following packages is require on the host.
```
# Add NVIDIA's apt repository and install the toolkit:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Create the CDI spec and set up Docker:
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# test by checking
nvidia-ctk cdi list          # should list nvidia.com/gpu=0 and nvidia.com/gpu=all
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

# docker image build
```
docker build -f .devcontainer/Dockerfile -t jasonwee/spam-model:gpu-nvidia ./.devcontainer/
```

# docker image push to dockerhub
```
docker push jasonwee/spam-model:gpu-nvidia
```

# check image is okay and gpu can be list
```
docker run --gpus all -it -v "$(pwd)":/workspace -w /workspace tflite-gpu bash
python -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"
```
