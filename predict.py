from read_vector import vector

import tensorflow as tf
import torch
import numpy as np

with open("model.tflite", "rb") as f:
        tflite_model = f.read()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

input_data = tf.constant(vector[0:10], dtype=tf.float32)
print(tf.shape(input_data))

print(input_data.shape)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

print(output_data)