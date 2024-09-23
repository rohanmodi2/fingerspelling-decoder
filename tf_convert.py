# from tf_model import HMMModel
from tensor_model import HMMModel
from read_hmm import states_, prior_probs_, transition_probs_, emission_paras_, vector
import tensorflow as tf

model = HMMModel(states=states_, prior_probs=prior_probs_, transition_probs=transition_probs_, emission_paras=emission_paras_)

input_shape = (None, 20)
model.build(input_shape)

most_likely_sequence1 = model(evidence_vector=vector[0:2])

model.export("saved_model")

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("complete")