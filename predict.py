from read_vector import vector

from ai_edge_litert.interpreter import Interpreter
import tensorflow as tf
import torch
import numpy as np
import time


def tf_predict(evidence_vector):
  model = tf.keras.layers.TFSMLayer('saved_model', call_endpoint='serving_default')
  predictions = model(tf.constant([evidence_vector], dtype=tf.float32))
  return predictions
  

def tf_convert():
  converter = tf.lite.TFLiteConverter.from_saved_model('saved_model') # path to the SavedModel directory
  converter.experimental_new_converter = True
  converter.allow_custom_ops = True
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.float32]
  converter.experimental_fixed_input_shape = [10, 20]  # For a fixed shape

  tflite_model = converter.convert()

  with open('model1.tflite', 'wb') as f:
    f.write(tflite_model)


def tflite_predict(evidence_vector):
  with open("model1.tflite", "rb") as f:
    tflite_model = f.read()

  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  input_data = tf.constant([evidence_vector], dtype=tf.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  return output_data


evidence_vector = vector[:40]
# print(tf_predict(evidence_vector))
tf_convert()
time_ = time.time()
print(tf.config.list_physical_devices('CPU'))
print(tflite_predict(evidence_vector))
print(tf.config.list_physical_devices('GPU'))
print(time.time() - time_)