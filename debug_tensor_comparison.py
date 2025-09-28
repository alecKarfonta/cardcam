import cv2
import numpy as np
import onnxruntime as ort

# Load and preprocess image exactly like JavaScript
img = cv2.imread('cam.png')
input_size = 1088

# Letterbox resize
scale = min(input_size / img.shape[1], input_size / img.shape[0])
new_width = int(img.shape[1] * scale)
new_height = int(img.shape[0] * scale)
pad_x = (input_size - new_width) // 2
pad_y = (input_size - new_height) // 2

resized = cv2.resize(img, (new_width, new_height))
padded = cv2.copyMakeBorder(resized, pad_y, input_size-new_height-pad_y, 
                           pad_x, input_size-new_width-pad_x, 
                           cv2.BORDER_CONSTANT, value=(128, 128, 128))

# Convert to tensor format [1, 3, H, W] normalized to [0,1]
tensor = padded.astype(np.float32) / 255.0
tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
tensor = np.expand_dims(tensor, axis=0)   # Add batch dimension

print(f'Input tensor shape: {tensor.shape}')
print(f'Input tensor dtype: {tensor.dtype}')
print(f'Input tensor range: [{tensor.min():.6f}, {tensor.max():.6f}]')

# Sample some pixel values to compare with JavaScript
print(f'Sample pixels (first 10 from R channel): {tensor[0, 0, 0, :10]}')
print(f'Sample pixels (center area): {tensor[0, 0, 544, 540:550]}')

# Save tensor for comparison
np.save('python_input_tensor.npy', tensor)
print('Saved input tensor to python_input_tensor.npy')

# Run inference
session = ort.InferenceSession('frontend/public/models/trading_card_detector_backbone_opset18.onnx', 
                              providers=['CPUExecutionProvider'])
outputs = session.run(None, {'images': tensor})
output = outputs[0]

print(f'Output shape: {output.shape}')
print(f'Output range: [{output.min():.6f}, {output.max():.6f}]')

# Save output for comparison
np.save('python_output_tensor.npy', output)
print('Saved output tensor to python_output_tensor.npy')

# Check specific values that JavaScript reported
print(f'First 10 cx values: {output[0, 0, :10]}')
print(f'First 10 cy values: {output[0, 1, :10]}')
print(f'First 10 confidence values: {output[0, 5, :10]}')

# Check the high-confidence detection we found
high_conf_idx = 10782
if high_conf_idx < output.shape[2]:
    cx, cy, w, h, angle, conf_raw = output[0, :, high_conf_idx]
    conf = 1 / (1 + np.exp(-conf_raw))
    print(f'High confidence detection at idx {high_conf_idx}:')
    print(f'  cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}, angle={angle:.6f}, conf={conf:.3f}')
