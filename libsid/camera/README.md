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

### License

This project is licensed under the MIT License.