from tflite_model_maker import ExportFormat, model_spec, text_classifier
from tflite_model_maker.text_classifier import DataLoader

import tensorflow as tf
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')

data_file = tf.keras.utils.get_file(fname='comment-spam.csv', origin='https://raw.githubusercontent.com/jasonwee/MyMachineLearningArtificialIntelligence/refs/heads/master/courses/build_a_comment_spam_machine_learning_model/lmblog_comments.csv', extract=False)

# This picks the model type. average_word_vec is the simplest text classifier
# in Model Maker. Inside, it's this Keras model:
# words → Embedding → GlobalAveragePooling1D → Dense(relu) → Dropout(0.2) → Dense(2, softmax)
spec = model_spec.get(spec_or_str='average_word_vec')
# This is the vocabulary size. Training counts every word and keeps the 2,000
# most frequent ones. Any other word becomes <UNKNOWN> and contributes no
# information.
spec.num_words = 2000 # library default: 10000
# This fixes how many words the model reads per comment. Every comment is
# converted to exactly 20 numbers: a <START> marker, then the words.
# Shorter comments are padded with <PAD>, and longer ones are cut off after
# word 19.
spec.seq_len = 20  # library default: 256
# This is the embedding size: how many numbers represent each word. It
# also sets the size of the hidden Dense layer. More dimensions let the model
# capture more nuance about each word, at the cost of more parameters to learn.
spec.wordvec_dim = 7 # library default: 16
spec.model_dir = f'./logs/w{spec.num_words}_s{spec.seq_len}_d{spec.wordvec_dim}'

data = DataLoader.from_csv(
    filename=data_file,
    text_column='commenttext',
    label_column='spam',
    model_spec=spec,
    delimiter=',',
    shuffle=True,
    is_training=True)

train_data, test_data = data.split(0.9)

# Build the model
model = text_classifier.create(
    train_data=train_data,
    model_spec=spec,
    epochs=50, 
    validation_data=test_data
)

# evaluating the model
loss, accuracy = model.evaluate(test_data)
print(f"Test loss: {loss:.4f} Test accuracy: {accuracy:.4f}")

model.export(export_dir='./mm_spam/')
model.export(
    export_dir='./mm_spam/', 
    export_format=[ExportFormat.LABEL, ExportFormat.VOCAB]
)
