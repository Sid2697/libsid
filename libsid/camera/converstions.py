import numpy as np


class PyTorchOpen3D:
    """
    Initialize the PyTorchOpen3D class with predefined transformation matrices.
    Open3D Coordinate System:

                z
                ↑
                /   
                /  
            /   
            +------→ x
            |
            |
            |
            y

    PyTorch3D Coordinate System:

            y
            ↑    z
            |   /
            |  /
            | / 
    x ------+ 

    These class is to convert between the two coordinate systems.
    """
    transform_4x4 = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    transform_3x3 = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ])

    @staticmethod
    def convert_4x4(pose: np.ndarray) -> np.ndarray:
        """
        Convert a 4x4 pose matrix using the predefined transformation matrix.

        Args:
            pose (np.ndarray): The 4x4 pose matrix to be transformed.

        Returns:
            np.ndarray: The transformed 4x4 pose matrix.

        Raises:
            ValueError: If pose is not a 4x4 matrix.
            TypeError: If pose is not a numpy array.
        """
        if not isinstance(pose, np.ndarray):
            raise TypeError(f'Expected pose as numpy array, got {type(pose)}')
        if pose.shape != (4, 4):
            raise ValueError(f'pose must be a 4x4 matrix, got {pose.shape}')
        return PyTorchOpen3D.transform_4x4 @ pose
    
    @staticmethod
    def convert_3x3(R: np.ndarray, T: np.ndarray) -> tuple:
        """
        Convert a 3x3 rotation matrix and a 3x1 translation vector using the predefined transformation matrix.

        Args:
            R (np.ndarray): The 3x3 rotation matrix to be transformed.
            T (np.ndarray): The 3x1 translation vector to be transformed.

        Returns:
            tuple: The transformed 3x3 rotation matrix and 3x1 translation vector.

        Raises:
            ValueError: If R is not a 3x3 matrix or T is not a 3x1 vector.
            TypeError: If R or T are not numpy arrays.
        """
        if not isinstance(R, np.ndarray):
            raise TypeError(f'Expected Rotation (R) as numpy array, got {type(R)}')
        if not isinstance(T, np.ndarray):
            raise TypeError(f'Expected Translation (T) as numpy array, got {type(T)}')
        if R.shape != (3, 3):
            raise ValueError(f'Rotation matrix must be a 3x3 matrix, got {R.shape}')
        if T.shape == (3, 1):
            T = T.reshape(3)
        elif T.shape != (3,):
            raise ValueError(f'Translation vector must be a 3x1 matrix, got {T.shape}')
        return PyTorchOpen3D.transform_3x3 @ R, PyTorchOpen3D.transform_3x3 @ T
