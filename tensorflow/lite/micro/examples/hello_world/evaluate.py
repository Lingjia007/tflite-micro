# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

import os
import matplotlib.pyplot as plt
from absl import app
from absl import flags
from absl import logging
import numpy as np
from tflite_micro.python.tflite_micro import runtime

OpResolverType = None
try:
  import ai_edge_litert.interpreter as tflite_interp
  from ai_edge_litert.interpreter import OpResolverType
except ImportError:
  try:
    import tflite_runtime.interpreter as tflite_interp
    from tflite_runtime.interpreter import OpResolverType
  except ImportError:
    try:
      import tensorflow.lite as tflite_interp
      from tensorflow.lite.experimental import OpResolverType
    except ImportError:
      raise ImportError(
          "Could not import ai_edge_litert, tflite_runtime, or tensorflow.")

_USE_TFLITE_INTERPRETER = flags.DEFINE_bool(
    'use_tflite',
    False,
    'Inference with the TF Lite interpreter instead of the TFLM interpreter',
)

_SAVE_PLOT = flags.DEFINE_bool(
    'save_plot',
    False,
    'Save the plot to file instead of displaying it',
)

_PREFIX_PATH = os.path.dirname(__file__)


def invoke_tflm_interpreter(input_shape, interpreter, x_value, input_index,
                            output_index):
  input_data = np.reshape(x_value, input_shape)
  interpreter.set_input(input_data, input_index)
  interpreter.invoke()
  y_quantized = np.reshape(interpreter.get_output(output_index), -1)[0]
  return y_quantized


def invoke_tflite_interpreter(input_shape, interpreter, x_value, input_index,
                              output_index):
  input_data = np.reshape(x_value, input_shape)
  interpreter.set_tensor(input_index, input_data)
  interpreter.invoke()
  tflite_output = interpreter.get_tensor(output_index)
  y_quantized = np.reshape(tflite_output, -1)[0]
  return y_quantized


# Generate a list of 1000 random floats in the range of 0 to 2*pi.
def generate_random_int8_input(sample_count=1000):
  # Generate a uniformly distributed set of random numbers in the range from
  # 0 to 2π, which covers a complete cosine wave oscillation
  np.random.seed(42)
  x_values = np.random.uniform(low=0, high=2 * np.pi,
                               size=sample_count).astype(np.int8)
  return x_values


# Generate a list of 1000 random floats in the range of 0 to 2*pi.
def generate_random_float_input(sample_count=1000):
  # Generate a uniformly distributed set of random numbers in the range from
  # 0 to 2π, which covers a complete cosine wave oscillation
  np.random.seed(42)
  x_values = np.random.uniform(low=0, high=2 * np.pi,
                               size=sample_count).astype(np.float32)
  return x_values


# Invoke the tflm interpreter with x_values in the range of [0, 2*PI] and
# returns the prediction of the interpreter.
def get_tflm_prediction(model_path, x_values):
  # Create the tflm interpreter
  tflm_interpreter = runtime.Interpreter.from_file(model_path)

  input_details = tflm_interpreter.get_input_details(0)
  output_details = tflm_interpreter.get_output_details(0)
  input_shape = np.array(input_details.get('shape'))
  input_type = input_details.get('dtype', np.float32)

  y_predictions = np.empty(x_values.size, dtype=np.float32)

  for i, x_value in enumerate(x_values):
    if input_type == np.int8:
      # Quantize float input to int8
      input_scale = input_details['quantization_parameters']['scales'][0]
      input_zero_point = input_details['quantization_parameters']['zero_points'][0]
      int8_val = int(round(x_value / input_scale + input_zero_point))
      int8_val = max(-128, min(127, int8_val))
      input_data = np.reshape(np.array(int8_val, dtype=np.int8), input_shape)
    else:
      input_data = np.reshape(x_value, input_shape)

    tflm_interpreter.set_input(input_data, 0)
    tflm_interpreter.invoke()
    output = np.reshape(tflm_interpreter.get_output(0), -1)[0]

    if input_type == np.int8:
      # Dequantize int8 output to float
      output_scale = output_details['quantization_parameters']['scales'][0]
      output_zero_point = output_details['quantization_parameters']['zero_points'][0]
      y_predictions[i] = (int(output) - output_zero_point) * output_scale
    else:
      y_predictions[i] = output

  return y_predictions


# Invoke the tflite interpreter with x_values in the range of [0, 2*PI] and
# returns the prediction of the interpreter.
def get_tflite_prediction(model_path, x_values):
  # TFLite interpreter
  kwargs = {"model_path": model_path}
  if OpResolverType is not None:
    kwargs["experimental_op_resolver_type"] = OpResolverType.BUILTIN_REF
  else:
    logging.warning(
        "Could not find OpResolverType. Reference kernels might not be used.")

  tflite_interpreter = tflite_interp.Interpreter(**kwargs)
  tflite_interpreter.allocate_tensors()

  input_details = tflite_interpreter.get_input_details()[0]
  output_details = tflite_interpreter.get_output_details()[0]
  input_shape = np.array(input_details.get('shape'))
  input_type = input_details.get('dtype', np.float32)

  y_predictions = np.empty(x_values.size, dtype=np.float32)

  for i, x_value in enumerate(x_values):
    if input_type == np.int8:
      # Quantize float input to int8
      input_scale = input_details['quantization_parameters']['scales'][0]
      input_zero_point = input_details['quantization_parameters']['zero_points'][0]
      int8_val = int(round(x_value / input_scale + input_zero_point))
      int8_val = max(-128, min(127, int8_val))
      input_data = np.reshape(np.array(int8_val, dtype=np.int8), input_shape)
    else:
      input_data = np.reshape(x_value, input_shape)

    tflite_interpreter.set_tensor(input_details['index'], input_data)
    tflite_interpreter.invoke()
    output = tflite_interpreter.get_tensor(output_details['index'])

    if input_type == np.int8:
      # Dequantize int8 output to float
      output_scale = output_details['quantization_parameters']['scales'][0]
      output_zero_point = output_details['quantization_parameters']['zero_points'][0]
      y_predictions[i] = (int(output.flatten()[0]) - output_zero_point) * output_scale
    else:
      y_predictions[i] = np.reshape(output, -1)[0]

  return y_predictions


def get_model_info(model_path):
  """Get model information: size, input/output details, quantization params."""
  model_size = os.path.getsize(model_path)
  info = {'size': model_size, 'size_kb': model_size / 1024.0}

  # Use TFLM to get input/output details
  tflm_interpreter = runtime.Interpreter.from_file(model_path)
  input_details = tflm_interpreter.get_input_details(0)
  output_details = tflm_interpreter.get_output_details(0)

  info['input_shape'] = input_details.get('shape', [])
  info['input_dtype'] = str(input_details.get('dtype', '?'))
  info['output_shape'] = output_details.get('shape', [])
  info['output_dtype'] = str(output_details.get('dtype', '?'))

  quant_in = input_details.get('quantization_parameters', {})
  quant_out = output_details.get('quantization_parameters', {})

  if quant_in.get('scales') is not None and len(quant_in['scales']) > 0:
    info['input_scale'] = quant_in['scales'][0]
    info['input_zero_point'] = quant_in['zero_points'][0]
  if quant_out.get('scales') is not None and len(quant_out['scales']) > 0:
    info['output_scale'] = quant_out['scales'][0]
    info['output_zero_point'] = quant_out['zero_points'][0]

  return info


def main(_):
  float_model_path = os.path.join(_PREFIX_PATH,
                                  'models/hello_world_float.tflite')
  int8_model_path = os.path.join(_PREFIX_PATH,
                                 'models/hello_world_int8.tflite')

  x_values = generate_random_float_input()

  # Calculate the corresponding cosine values
  y_true_values = np.cos(x_values).astype(np.float32)

  # Get model info
  float_info = get_model_info(float_model_path)
  int8_info = get_model_info(int8_model_path)

  fig, axes = plt.subplots(1, 2, figsize=(16, 6))

  # --- Float model plot ---
  ax1 = axes[0]
  float_title = (f"Float Model  (size: {float_info['size_kb']:.1f} KB)\n"
                 f"input: {float_info['input_dtype']} {float_info['input_shape']}, "
                 f"output: {float_info['output_dtype']} {float_info['output_shape']}")
  ax1.set_title(float_title, fontsize=10)
  ax1.set_xlabel('Input (x)')
  ax1.set_ylabel('Output (y)')

  if _USE_TFLITE_INTERPRETER.value:
    y_float_pred = get_tflite_prediction(float_model_path, x_values)
    ax1.plot(x_values, y_float_pred, 'b.', label='TFLite Float', markersize=2)
    logging.info('TFLite float predictions: min=%f, max=%f, mean=%f',
                 np.min(y_float_pred), np.max(y_float_pred),
                 np.mean(y_float_pred))
  else:
    y_float_pred = get_tflm_prediction(float_model_path, x_values)
    ax1.plot(x_values, y_float_pred, 'b.', label='TFLM Float', markersize=2)
    logging.info('TFLM float predictions: min=%f, max=%f, mean=%f',
                 np.min(y_float_pred), np.max(y_float_pred),
                 np.mean(y_float_pred))

  # Compute MSE for float model
  float_mse = np.mean((y_true_values - y_float_pred) ** 2)
  ax1.plot(x_values, y_true_values, 'r.', label='Actual Cosine', markersize=2)
  ax1.legend(loc='upper right', fontsize=8)
  ax1.grid(True, alpha=0.3)
  ax1.text(0.02, 0.02, f'MSE: {float_mse:.6f}', transform=ax1.transAxes,
           fontsize=9, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

  # --- Int8 quantized model plot ---
  ax2 = axes[1]
  int8_title = (f"Int8 Quantized Model  (size: {int8_info['size_kb']:.1f} KB)\n"
                f"input: {int8_info['input_dtype']} {int8_info['input_shape']}, "
                f"output: {int8_info['output_dtype']} {int8_info['output_shape']}")
  if 'input_scale' in int8_info:
    int8_title += (f"\nscale: in={int8_info['input_scale']:.4f}/"
                   f"out={int8_info['output_scale']:.4f}, "
                   f"zp: in={int8_info['input_zero_point']}/"
                   f"out={int8_info['output_zero_point']}")
  ax2.set_title(int8_title, fontsize=10)
  ax2.set_xlabel('Input (x)')
  ax2.set_ylabel('Output (y)')

  if _USE_TFLITE_INTERPRETER.value:
    y_int8_pred = get_tflite_prediction(int8_model_path, x_values)
    ax2.plot(x_values, y_int8_pred, 'g.', label='TFLite Int8', markersize=2)
    logging.info('TFLite int8 predictions: min=%f, max=%f, mean=%f',
                 np.min(y_int8_pred), np.max(y_int8_pred),
                 np.mean(y_int8_pred))
  else:
    y_int8_pred = get_tflm_prediction(int8_model_path, x_values)
    ax2.plot(x_values, y_int8_pred, 'g.', label='TFLM Int8', markersize=2)
    logging.info('TFLM int8 predictions: min=%f, max=%f, mean=%f',
                 np.min(y_int8_pred), np.max(y_int8_pred),
                 np.mean(y_int8_pred))

  # Compute MSE for int8 model
  int8_mse = np.mean((y_true_values - y_int8_pred) ** 2)
  ax2.plot(x_values, y_true_values, 'r.', label='Actual Cosine', markersize=2)
  ax2.legend(loc='upper right', fontsize=8)
  ax2.grid(True, alpha=0.3)
  ax2.text(0.02, 0.02, f'MSE: {int8_mse:.6f}', transform=ax2.transAxes,
           fontsize=9, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

  logging.info('True cosine values: min=%f, max=%f, mean=%f',
               np.min(y_true_values), np.max(y_true_values),
               np.mean(y_true_values))
  logging.info('Float model MSE: %f, Int8 model MSE: %f', float_mse, int8_mse)

  plt.tight_layout()

  # Either save or display the plot
  if _SAVE_PLOT.value:
    output_dir = '/tmp'
    if _USE_TFLITE_INTERPRETER.value:
      output_file = os.path.join(output_dir, 'hello_world_tflite_cosine.png')
    else:
      output_file = os.path.join(output_dir, 'hello_world_tflm_cosine.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    logging.info('Plot saved to %s', output_file)
  else:
    plt.show()


if __name__ == '__main__':
  app.run(main)
