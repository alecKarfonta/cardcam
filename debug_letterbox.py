import cv2
import numpy as np

# Load the same image
img = cv2.imread('cam.png')
print(f'Original image: {img.shape}')  # (1080, 1920, 3)

input_size = 1088

# Python letterbox (what we just tested)
scale = min(input_size / img.shape[1], input_size / img.shape[0])  # width, height
new_width = int(img.shape[1] * scale)
new_height = int(img.shape[0] * scale)
pad_x = (input_size - new_width) // 2
pad_y = (input_size - new_height) // 2

print(f'Python letterbox: scale={scale:.3f}, new_size={new_width}x{new_height}, pad={pad_x},{pad_y}')

# JavaScript letterbox (from our debug output)
js_scale = min(input_size / 1920, input_size / 1080)  # imageWidth, imageHeight
js_new_width = round(1920 * js_scale)
js_new_height = round(1080 * js_scale)
js_pad_x = (input_size - js_new_width) // 2
js_pad_y = (input_size - js_new_height) // 2

print(f'JavaScript letterbox: scale={js_scale:.3f}, new_size={js_new_width}x{js_new_height}, pad={js_pad_x},{js_pad_y}')

# Check if they match
print(f'Scale match: {abs(scale - js_scale) < 0.001}')
print(f'Dimensions match: {new_width == js_new_width and new_height == js_new_height}')
print(f'Padding match: {pad_x == js_pad_x and pad_y == js_pad_y}')

# The detection coordinates should be:
# Python: cx=311.2, cy=635.6
# Let's see where this maps in the original image

# Convert model coordinates back to original image
def model_to_image_coords(model_x, model_y, scale, pad_x, pad_y):
    img_x = (model_x - pad_x) / scale
    img_y = (model_y - pad_y) / scale
    return img_x, img_y

python_img_x, python_img_y = model_to_image_coords(311.2, 635.6, scale, pad_x, pad_y)
js_img_x, js_img_y = model_to_image_coords(345.0, 320.9, js_scale, js_pad_x, js_pad_y)

print(f'\nCoordinate conversion:')
print(f'Python model (311.2, 635.6) -> image ({python_img_x:.1f}, {python_img_y:.1f})')
print(f'JavaScript model (345.0, 320.9) -> image ({js_img_x:.1f}, {js_img_y:.1f})')

print(f'\nAs percentages of image:')
print(f'Python: ({python_img_x/1920*100:.1f}%, {python_img_y/1080*100:.1f}%)')
print(f'JavaScript: ({js_img_x/1920*100:.1f}%, {js_img_y/1080*100:.1f}%)')
