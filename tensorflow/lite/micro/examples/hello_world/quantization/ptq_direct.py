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
"""Quantize the hello_world model directly from Keras model.

This script trains the model and directly converts it to a quantized TFLite model
without requiring an intermediate SavedModel format.

Run:
`bazel build tensorflow/lite/micro/examples/hello_world/quantization:ptq_direct`
`bazel-bin/tensorflow/lite/micro/examples/hello_world/quantization/ptq_direct --target_dir=/tmp/quant_model/`
"""
import math
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("target_dir", "/tmp/quant_model",
                    "The directory to save the quantized model.")
flags.DEFINE_integer("epochs", 500, "Number of epochs to train the model.")


def get_data():
  """Generate training data for cosine function."""
  x_values = np.random.uniform(low=0, high=2 * math.pi,
                               size=1000).astype(np.float32)
  np.random.shuffle(x_values)
  y_values = np.cos(x_values).astype(np.float32)
  return x_values, y_values


def create_model() -> tf.keras.Model:
  """Create the hello_world model."""
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(1, )))
  model.add(tf.keras.layers.Dense(16, activation='relu'))
  model.add(tf.keras.layers.Dense(1))
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])
  return model


def convert_to_quantized_tflite(model, x_values):
  """Convert Keras model to quantized TFLite model.

  Args:
      model: Trained Keras model
      x_values: Representative dataset for quantization

  Returns:
      Quantized TFLite model in serialized format.
  """
  def representative_dataset(num_samples=500):
    for i in range(num_samples):
      yield [x_values[i].reshape(1, 1)]

  # Convert from Keras model directly
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8
  converter.inference_output_type = tf.int8
  converter.representative_dataset = representative_dataset
  tflite_model = converter.convert()
  return tflite_model


def save_tflite_model(tflite_model, target_dir, model_name):
  """Save the TFLite model to disk."""
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)
  save_path = os.path.join(target_dir, model_name)
  with open(save_path, "wb") as f:
    f.write(tflite_model)
  logging.info("Quantized TFLite model saved to %s", save_path)


def main(_):
  # Generate training data
  x_values, y_values = get_data()

  # Create and train the model
  model = create_model()
  model.fit(x_values, y_values,
            epochs=FLAGS.epochs,
            validation_split=0.2,
            batch_size=64,
            verbose=2)

  # Convert to quantized TFLite
  quantized_model = convert_to_quantized_tflite(model, x_values)

  # Save the model
  save_tflite_model(quantized_model, FLAGS.target_dir, "hello_world_int8.tflite")


if __name__ == "__main__":
  app.run(main)