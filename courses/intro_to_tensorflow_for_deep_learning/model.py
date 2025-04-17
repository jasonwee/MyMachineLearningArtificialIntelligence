import kagglehub

# Download latest version
path = kagglehub.model_download("tensorflow/spam-detection/tensorFlow2/tutorials-spam-detection")

print("Path to model files:", path)
