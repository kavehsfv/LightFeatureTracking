
import cv2
import numpy as np
import torch
from typing import Callable, List, Optional, Tuple, Union

def get_key_by_value(my_dict, search_value):
    for key, value in my_dict.items():
        if value == search_value:
            return key
    return None

def replace_key(dictionary, old_key, new_key):
    if old_key in dictionary:
        dictionary[new_key] = dictionary.pop(old_key)

def remove_key(dictionary, key):
    if key in dictionary:
        dictionary.pop(key)
        return True
    else:
        return False

def tensor_to_tuple(tensor):
    # If the tensor is a tuple containing the frame number and the tensor
    if isinstance(tensor, tuple):
        frame_no, point = tensor
        return (frame_no, (point[0].item(), point[1].item()))
    # If the tensor is just a tensor without a frame number
    else:
        return (tensor[0].item(), tensor[1].item())

def frame_exists(keypoints, frame_number):
    return any(f == frame_number for f, _ in keypoints)

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

def load_data_image_crop(image_data: bytes, resize: Optional[Union[int, Tuple[int, int]]] = None, crop: Optional[Tuple[int, int, int, int]] = None, **kwargs) -> torch.Tensor:
    """Process and crop the image as needed, then convert to a torch.Tensor"""

    # Convert bytes data to a NumPy array
    image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    imageCV = image_array
    # Crop the image if a crop section is provided
    if crop is not None:
        x1, y1, x2, y2 = crop
        image_array = image_array[y1:y2, x1:x2]

    # Resize the image if needed
    if resize is not None:
        if isinstance(resize, int):
            resize = (resize, resize)  # Assuming square resize if only one dimension is provided
        image_array = cv2.resize(image_array, resize)

    # Convert the NumPy array to a torch.Tensor
    return numpy_image_to_torch(image_array), imageCV[..., ::-1]


def load_data_image_crop_mdfy(
    image_data: bytes,
    resize: Optional[Union[int, Tuple[int, int]]] = None,
    crop: Optional[Tuple[int, int, int, int]] = None,
    **kwargs
) -> torch.Tensor:
    """Process and crop the image as needed, then convert to a torch.Tensor"""

    # Convert bytes data to a NumPy array
    image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    
    # **Add this line to convert BGR to RGB**
    image_array = image_array[..., ::-1]

    # Make a copy for imageCV after color conversion
    imageCV = image_array.copy()

    # Crop the image if a crop section is provided
    if crop is not None:
        x1, y1, x2, y2 = crop
        image_array = image_array[y1:y2, x1:x2]

    # Resize the image if needed
    if resize is not None:
        if isinstance(resize, int):
            resize = (resize, resize)  # Assuming square resize if only one dimension is provided
        image_array = cv2.resize(image_array, resize)

    # Convert the NumPy array to a torch.Tensor
    return numpy_image_to_torch(image_array), imageCV