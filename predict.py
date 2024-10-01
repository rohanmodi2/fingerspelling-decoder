from read_vector import vector

import tensorflow as tf
import torch
import numpy as np
import time


model = tf.keras.layers.TFSMLayer('saved_model', call_endpoint='serving_default')
predictions = model(tf.constant([vector[0:10]], dtype=tf.float32))
time_ = time.time()
print("tensorflow output:", predictions)
print("time", time.time() - time_)
input("done")

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model') # path to the SavedModel directory
converter.experimental_new_converter = True
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
converter.experimental_fixed_input_shape = [10, 20]  # For a fixed shape

tflite_model = converter.convert()

# Save the model.
with open('model1.tflite', 'wb') as f:
  f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="model1.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
input_shape = [10, 20]
interpreter.resize_tensor_input(input_details[0]['index'], input_shape)
interpreter.allocate_tensors()
input_data = np.random.rand(10, 20).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

print(input_details)

input("converted to tflite")


from ai_edge_litert.interpreter import Interpreter
interpreter = Interpreter('model1.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
print('input_details:')
print(input_details)
input_data = tf.constant([vector[0:10]], dtype=tf.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print("tflite output:", output_data)



import sys
sys.exit()
input()
my_signature = interpreter.get_signature_runner()
output = my_signature(evidence_vector=tf.constant(vector[0:10], dtype=tf.float32))
print("tflite output:", output)



# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

print(input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

input("done")
model = tf.saved_model.load("saved_model")
print(list(model.signatures.keys()))
infer = model.signatures["serving_default"]
print(infer.structured_outputs)
labeling = infer(tf.constant(x))[pretrained_model.output_names[0]]




result = model(evidence_vector=torch.tensor(vector[0:20]))



print(result)







with open("model.tflite", "rb") as f:
        tflite_model = f.read()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

input_data = tf.constant(vector[0:1], dtype=tf.float32)
print(tf.shape(input_data))

print(input_data.shape)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

print(output_data)