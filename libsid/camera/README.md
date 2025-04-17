# Camera Module

This module provides a `CameraManager` class to manage camera parameters and transformations. It includes methods to set up the camera with world-to-camera and camera intrinsics matrices, and to retrieve these matrices.

## Class: `CameraManager`

### Initialization

```python
CameraManager(camera_id: str)
```

**Args:**
- `camera_id` (str): The ID of the camera.

**Properties:**
- `camera_id`: Returns the ID of the camera.
- `w2c`: Returns the world-to-camera transformation matrix.
- `intrinsics`: Returns the camera intrinsics matrix.
- `c2w`: Returns the camera-to-world transformation matrix.

### Methods

#### `setup(w2c: np.ndarray, intrinsics: np.ndarray) -> None`

Sets up the camera with the given world-to-camera transformation matrix and intrinsics.

**Args:**
- `w2c` (np.ndarray): The world-to-camera transformation matrix.
- `intrinsics` (np.ndarray): The camera intrinsics matrix.

#### `set_w2c(w2c: np.ndarray) -> None`

Sets the world-to-camera transformation matrix.

**Args:**
- `w2c` (np.ndarray): The world-to-camera transformation matrix.

**Raises:**
- `ValueError`: If `w2c` is not a 4x4 matrix.
- `TypeError`: If `w2c` is not a numpy array.

#### `set_intrinsics(intrinsics: np.ndarray) -> None`

Sets the camera intrinsics matrix.

**Args:**
- `intrinsics` (np.ndarray): The camera intrinsics matrix.

**Raises:**
- `ValueError`: If `intrinsics` is not a 3x3 matrix.
- `TypeError`: If `intrinsics` is not a numpy array.

#### `set_c2w(c2w: np.ndarray = None) -> None`

Sets the camera-to-world transformation matrix. If not provided, it will be calculated as the inverse of `w2c`.

**Args:**
- `c2w` (np.ndarray, optional): The camera-to-world transformation matrix.

**Raises:**
- `ValueError`: If `w2c` is not set and `c2w` is not provided.
- `TypeError`: If `c2w` is not a numpy array.
- `ValueError`: If `c2w` is not a 4x4 matrix.

### Usage Example

```python
import numpy as np
from camera import CameraManager

camera_manager = CameraManager(camera_id="camera_1")
w2c_matrix = np.eye(4)
intrinsics_matrix = np.eye(3)

camera_manager.setup(w2c=w2c_matrix, intrinsics=intrinsics_matrix)
```

### Dependencies

- `numpy`

## Class: `PyTorchOpen3D`

### Description

The `PyTorchOpen3D` class provides methods to convert between the Open3D and PyTorch3D coordinate systems using predefined transformation matrices.

### Properties

- `transform_4x4`: A 4x4 transformation matrix for converting between coordinate systems.
- `transform_3x3`: A 3x3 transformation matrix for converting between coordinate systems.

### Methods

#### `convert_4x4(pose: np.ndarray) -> np.ndarray`

Converts a 4x4 pose matrix using the predefined transformation matrix.

**Args:**
- `pose` (np.ndarray): The 4x4 pose matrix to be transformed.

**Returns:**
- `np.ndarray`: The transformed 4x4 pose matrix.

**Raises:**
- `ValueError`: If `pose` is not a 4x4 matrix.
- `TypeError`: If `pose` is not a numpy array.

#### `convert_3x3(R: np.ndarray, T: np.ndarray) -> tuple`

Converts a 3x3 rotation matrix and a 3x1 translation vector using the predefined transformation matrix.

**Args:**
- `R` (np.ndarray): The 3x3 rotation matrix to be transformed.
- `T` (np.ndarray): The 3x1 translation vector to be transformed.

**Returns:**
- `tuple`: The transformed 3x3 rotation matrix and 3x1 translation vector.

**Raises:**
- `ValueError`: If `R` is not a 3x3 matrix or `T` is not a 3x1 vector.
- `TypeError`: If `R` or `T` are not numpy arrays.

### Usage Example

```python
import numpy as np
from camera.converstions import PyTorchOpen3D

pose_matrix = np.eye(4)
rotation_matrix = np.eye(3)
translation_vector = np.array([1, 2, 3])

transformed_pose = PyTorchOpen3D.convert_4x4(pose_matrix)
transformed_rotation, transformed_translation = PyTorchOpen3D.convert_3x3(rotation_matrix, translation_vector)
```

### Dependencies

- `numpy`
