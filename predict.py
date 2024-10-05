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
  converter.experimental_fixed_input_shape = [60, 20]  # For a fixed shape

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

  from read_hmm import states_
  result = []
  for n in output_data:
    result.append(states_[n])

  return result


def python_predict(evidence_vector):
  from read_hmm import states_, prior_probs_, transition_probs_, emission_paras_
  from decoder import multidimensional_viterbi
  return multidimensional_viterbi(evidence_vector, states_, prior_probs_,
                                  transition_probs_, emission_paras_, 20)
  


evidence_vector = vector[:60]
# print(tf_predict(evidence_vector))
# tf_convert()
time_ = time.time()
print("tflite output: \n", tflite_predict(evidence_vector))
print("python output: \n", python_predict(evidence_vector))
# print(tf.config.list_physical_devices('CPU'))
# print(tf.config.list_physical_devices('GPU'))
print(time.time() - time_)