# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""hello_world model training for wave recognition

Run:
`bazel build tensorflow/lite/micro/examples/hello_world:train`

Train cosine model (default):
`bazel-bin/tensorflow/lite/micro/examples/hello_world/train --save_tf_model --save_dir=/tmp/model_created/`

Train sine model:
`bazel-bin/tensorflow/lite/micro/examples/hello_world/train --save_tf_model --save_dir=/tmp/model_created/ --function_type=sin`

Customize epochs:
`bazel-bin/tensorflow/lite/micro/examples/hello_world/train --save_tf_model --save_dir=/tmp/model_created/ --epochs=2000`
"""
import math
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", 500, "number of epochs to train the model.")
flags.DEFINE_string("save_dir", "/tmp/hello_world_models",
                    "the directory to save the trained model.")
flags.DEFINE_boolean("save_tf_model", False,
                     "store the original unconverted tf model.")
flags.DEFINE_string("function_type", "cos",
                    "the function to train: 'cos' for cosine, 'sin' for sine. Default is 'cos'.")


def get_data():
  """
  The code will generate a set of random `x` values, calculate their cosine
  or sine values based on function_type flag.
  """
  # Generate a uniformly distributed set of random numbers in the range from
  # 0 to 2π, which covers a complete wave oscillation
  x_values = np.random.uniform(low=0, high=2 * math.pi,
                               size=1000).astype(np.float32)

  # Shuffle the values to guarantee they're not in order
  np.random.shuffle(x_values)

  # Calculate the corresponding function values based on function_type
  if FLAGS.function_type == "sin":
    y_values = np.sin(x_values).astype(np.float32)
  else:  # Default is cosine
    y_values = np.cos(x_values).astype(np.float32)

  return (x_values, y_values)


def create_model() -> tf.keras.Model:
  model = tf.keras.Sequential()

  # First layer takes a scalar input and feeds it through 32 "neurons". The
  # neurons decide whether to activate based on the 'relu' activation function.
  # Increased from 16 to 32 neurons for better learning capacity.
  model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(1, )))

  # Second layer with 32 neurons to help the network learn more complex
  # representations. Increased from 16 to 32 neurons.
  model.add(tf.keras.layers.Dense(32, activation='relu'))

  # Third layer with 16 neurons for additional complexity
  model.add(tf.keras.layers.Dense(16, activation='relu'))

  # Final layer is a single neuron, since we want to output a single value
  model.add(tf.keras.layers.Dense(1))

  # Compile the model using the standard 'adam' optimizer and the mean squared
  # error or 'mse' loss function for regression.
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])

  return model


def convert_tflite_model(model):
  """Convert the save TF model to tflite model, then save it as .tflite flatbuffer format
    Args:
        model (tf.keras.Model): the trained hello_world Model
    Returns:
        The converted model in serialized format.
  """
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  return tflite_model


def save_tflite_model(tflite_model, save_dir, model_name):
  """save the converted tflite model
  Args:
      tflite_model (binary): the converted model in serialized format.
      save_dir (str): the save directory
      model_name (str): model name to be saved
  """
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  save_path = os.path.join(save_dir, model_name)
  with open(save_path, "wb") as f:
    f.write(tflite_model)
  logging.info("Tflite model saved to %s", save_dir)


def train_model(epochs, x_values, y_values):
  """Train keras hello_world model
    Args: epochs (int) : number of epochs to train the model
        x_train (numpy.array): list of the training data
        y_train (numpy.array): list of the corresponding array
    Returns:
        tf.keras.Model: A trained keras hello_world model
  """
  model = create_model()
  model.fit(x_values,
            y_values,
            epochs=epochs,
            validation_split=0.2,
            batch_size=64,
            verbose=2)

  if FLAGS.save_tf_model:
    model.export(FLAGS.save_dir)
    logging.info("TF model saved to %s", FLAGS.save_dir)

  return model


def main(_):
  x_values, y_values = get_data()
  trained_model = train_model(FLAGS.epochs, x_values, y_values)

  # Convert and save the model to .tflite
  tflite_model = convert_tflite_model(trained_model)
  model_name = f"hello_world_{FLAGS.function_type}_float.tflite"
  save_tflite_model(tflite_model,
                    FLAGS.save_dir,
                    model_name=model_name)


if __name__ == "__main__":
  app.run(main)