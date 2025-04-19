import time
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from tensorflow.keras import layers
from tensorflow.keras.layers import Layer


# setup
# apt-get install libbz2-dev
# python3.11 becuase it has stdlib-python3.11 installed which contain libbz2
# python3.11 -m venv env_model_save
# source env_model_save/bin/activate
# pip install --upgrade pip
# pip install -U tensorflow_hub
# pip install -U tensorflow_datasets
# pip install matplotlib

(train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)

def format_image(image, label):
    # `hub` image modules expect their data normalized to the [0,1] range.
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
    return image, label

num_examples = info.splits['train'].num_examples

BATCH_SIZE = 32
IMAGE_RES = 224


# Dataset creation function
def create_dataset(dataset, batch_size, num_examples):
    return (
        dataset
        .cache()
        .shuffle(num_examples // 4)
        .map(format_image)
        .batch(batch_size)
        .prefetch(1)
    )

train_batches = create_dataset(train_examples, BATCH_SIZE, num_examples)
validation_batches = create_dataset(validation_examples, BATCH_SIZE, num_examples)


URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_RES, IMAGE_RES,3))

feature_extractor.trainable = False

class MyHubLayer(Layer):  # Define a custom Layer subclass
    def __init__(self, handle, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.handle = handle
        self.hub_layer = hub.KerasLayer(handle, trainable=trainable)

    def call(self, inputs):
        return self.hub_layer(inputs)

    def get_config(self):  # Needed for saving and loading
        config = super().get_config()
        config.update({
            "handle": self.handle,
            "trainable": self.hub_layer.trainable,
        })
        return config


# Use MyHubLayer instead of hub.KerasLayer directly
feature_extractor = MyHubLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))

model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(2)
])

model.summary()

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[train_accuracy])

EPOCHS = 1

# Custom training loop
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(labels, predictions)
    return loss

for epoch in range(EPOCHS):
    train_accuracy.reset_state()
    for images, labels in train_batches:
        loss = train_step(images, labels)
    print(f"Epoch: {epoch+1}, Loss: {loss:.4f}, Accuracy: {train_accuracy.result():.4f}")

class_names = np.array(info.features['label'].names)
print(class_names)

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()
predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]
print(predicted_class_names)

print("Labels: ", label_batch)
print("Predicted labels: ", predicted_ids)

plt.figure(figsize=(10, 9))
for n in range(30):
    plt.subplot(6, 5, n + 1)
    plt.imshow(image_batch[n])
    color = "blue" if predicted_ids[n] == label_batch[n] else "red"
    plt.title(predicted_class_names[n].title(), color=color)
    plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")

t = time.time()

export_path_keras = "./{}.h5".format(int(t))
print(export_path_keras)

model.save(export_path_keras)

reloaded = tf.keras.models.load_model(
    export_path_keras,
    # `custom_objects` tells keras how to load a `hub.KerasLayer`
    custom_objects={'KerasLayer': hub.KerasLayer, 'MyHubLayer': MyHubLayer} # Add MyHubLayer to custom_objects
)


reloaded.summary()


result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

print((abs(result_batch - reloaded_result_batch)).max())

# Custom training loop for reloaded model
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

reloaded.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[train_accuracy])
@tf.function
def train_step_reloaded(images, labels):
    with tf.GradientTape() as tape:
        predictions = reloaded(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, reloaded.trainable_variables)
    optimizer.apply_gradients(zip(gradients, reloaded.trainable_variables))
    train_accuracy.update_state(labels, predictions)
    return loss

for epoch in range(EPOCHS):
    train_accuracy.reset_state()
    for images, labels in train_batches:
        loss = train_step_reloaded(images, labels)
    print(f"Epoch: {epoch+1}, Loss: {loss:.4f}, Accuracy: {train_accuracy.result():.4f}")

t = time.time()

export_path_sm = "./{}".format(int(t))
print(export_path_sm)

tf.saved_model.save(model, export_path_sm)

"""
following codes does not work!!

reloaded_sm = tf.saved_model.load(export_path_sm)

reload_sm_result_batch = reloaded_sm(image_batch, training=False).numpy()

print((abs(result_batch - reload_sm_result_batch)).max())

t = time.time()

export_path_sm = "./{}".format(int(t))
print(export_path_sm)
tf.saved_model.save(model, export_path_sm)

reload_sm_keras = tf.keras.models.load_model(
  export_path_sm,
  custom_objects={'KerasLayer': hub.KerasLayer})

reload_sm_keras.summary()

result_batch = model.predict(image_batch)
reload_sm_keras_result_batch = reload_sm_keras.predict(image_batch)

print((abs(result_batch - reload_sm_keras_result_batch)).max())

"""
