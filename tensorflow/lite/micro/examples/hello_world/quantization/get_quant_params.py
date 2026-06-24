"""Get quantization parameters from the int8 model."""
import math
import numpy as np
import tensorflow as tf

model_path = '/home/lingsir/Projects/tflite-micro/tensorflow/lite/micro/examples/hello_world/models/hello_world_int8.tflite'

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale = input_details['quantization_parameters']['scales'][0]
input_zero_point = input_details['quantization_parameters']['zero_points'][0]
output_scale = output_details['quantization_parameters']['scales'][0]
output_zero_point = output_details['quantization_parameters']['zero_points'][0]

print(f"Input scale: {input_scale}")
print(f"Input zero_point: {input_zero_point}")
print(f"Output scale: {output_scale}")
print(f"Output zero_point: {output_zero_point}")

# Calculate int8 values for test inputs
golden_inputs_float = [0.77, 1.57, 2.3, 3.14]

int8_values = []
for val in golden_inputs_float:
    int8_val = int(round(val / input_scale + input_zero_point))
    int8_val = max(-128, min(127, int8_val))
    int8_values.append(int8_val)

print(f"\nInt8 golden values: {{{', '.join(str(v) for v in int8_values)}}}")

# Verify
for i, val in enumerate(golden_inputs_float):
    interpreter.set_tensor(input_details['index'], np.array([[int8_values[i]]], dtype=np.int8))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])
    y_pred = (output[0][0] - output_zero_point) * output_scale
    y_true = math.cos(val)
    diff = abs(y_true - y_pred)
    print(f"  input={val}, int8={int8_values[i]}, pred={y_pred:.6f}, true={y_true:.6f}, diff={diff:.6f}, pass={'Y' if diff <= 0.05 else 'N'}")
