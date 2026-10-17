import numpy as np
import tensorflow as tf
from tflite_model_maker import model_spec

# 1. Load the model configuration spec so we can reuse its text preprocessing
spec = model_spec.get('average_word_vec')
spec.num_words = 2000
spec.seq_len = 20

# 2. Load the vocabulary exported alongside the trained model
# (This is the word-to-number mapping so we can process raw sentences;
# it must come from the same training run as the model)
#spec.load_vocab("./mm_spam/vocab.txt")
spec.load_vocab("./output_model/vocab.txt")

# 3. Define the sentences you want to test
test_sentences = [
    "Check out this link to get free money fast!!!",  # Obvious Spam
    "Hey friend, I really enjoyed your latest video project.",  # Safe Comment
    "Subscribe to my channel for cheap deals today"  # Potential Spam
]

# 4. Load and spin up the TFLite Interpreter engine
#interpreter = tf.lite.Interpreter(model_path="./mm_spam/model.tflite")
interpreter = tf.lite.Interpreter(model_path="./output_model/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n=== Running Spam Detection Inference ===\n")

for sentence in test_sentences:
    # Step A: Tokenize and Pad the raw string into an array of exactly 20 integers
    processed_input = spec.preprocess(sentence)
    
    # Step B: Reshape the array to match the model's expected shape: [Batch_Size, Sequence_Length]
    input_data = np.array([processed_input], dtype=np.int32)
    
    # Step C: Feed the processed array into the model tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Step D: Run the mathematical prediction cycle
    interpreter.invoke()
    
    # Step E: Extract the output prediction array
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # The output returns two values: [Probability of False, Probability of True]
    # index 1 represents 'True' (Spam)
    spam_probability = output_data[1]
    
    # Step F: Interpret the probability score for the user
    is_spam = spam_probability > 0.5
    status = "🚨 SPAM DETECTED" if is_spam else "✅ SAFE COMMENT"
    
    print(f"Sentence: \"{sentence}\"")
    print(f"Score:     {spam_probability:.4f} ({spam_probability * 100:.1f}% confidence)")
    print(f"Verdict:   {status}\n")
