import numpy as np
from typing import Optional


class CameraManager():
    def __init__(
            self,
            camera_id: str,
            w2c: Optional[np.ndarray] = None,
            intrinsics: Optional[np.ndarray] = None,
    ):
        """
        Initialize the CameraManager with a camera ID.

        Args:
            camera_id (str): The ID of the camera.
        """
        self._camera_id = camera_id
        if w2c is not None:
            self.set_w2c(w2c)
            self.set_c2w()
        else:
            self._w2c = None
            self._c2w = None
        if intrinsics is not None:
            self.set_intrinsics(intrinsics)
        else:
            self._intrinsics = None
    
    @property
    def camera_id(self) -> str:
        """
        Get the camera ID.

        Returns:
            str: The ID of the camera.
        """
        return self._camera_id

    def setup(self, w2c: np.ndarray, intrinsics: np.ndarray) -> None:
        """
        Set up the camera with the given world-to-camera transformation matrix and intrinsics.

        Args:
            w2c (np.ndarray): The world-to-camera transformation matrix.
            intrinsics (np.ndarray): The camera intrinsics matrix.
        """
        self.set_w2c(w2c)
        self.set_c2w()
        self.set_intrinsics(intrinsics)
    
    def set_w2c(self, w2c: np.ndarray) -> None:
        """
        Set the world-to-camera transformation matrix.

        Args:
            w2c (np.ndarray): The world-to-camera transformation matrix.

        Raises:
            ValueError: If w2c is not a 4x4 matrix.
            TypeError: If w2c is not a numpy array.
        """
        if w2c.shape != (4, 4):
            raise ValueError(f'w2c must be a 4x4 matrix, got {w2c.shape}')
        if not isinstance(w2c, np.ndarray):
            raise TypeError(f'Expected w2c as numpy array, got {type(w2c)}')
        self._w2c = w2c
    
    @property
    def w2c(self) -> np.ndarray:
        """
        Get the world-to-camera transformation matrix.

        Returns:
            np.ndarray: The world-to-camera transformation matrix.

        Raises:
            AttributeError: If w2c is not set.
        """
        if self._w2c is None:
            raise AttributeError('w2c is not set')
        return self._w2c

    def set_intrinsics(self, intrinsics: np.ndarray) -> None:
        """
        Set the camera intrinsics matrix.

        Args:
            intrinsics (np.ndarray): The camera intrinsics matrix.

        Raises:
            ValueError: If intrinsics is not a 3x3 matrix.
            TypeError: If intrinsics is not a numpy array.
        """
        if not isinstance(intrinsics, np.ndarray):
            raise TypeError(f'Expected intrinsics as numpy array, got {type(intrinsics)}')
        if intrinsics.shape != (3, 3):
            raise ValueError(f'intrinsics must be a 3x3 matrix, got {intrinsics.shape}')
        self._intrinsics = intrinsics
    
    @property
    def intrinsics(self) -> np.ndarray:
        """
        Get the camera intrinsics matrix.

        Returns:
            np.ndarray: The camera intrinsics matrix.

        Raises:
            AttributeError: If intrinsics is not set.
        """
        if self._intrinsics is None:
            raise AttributeError('intrinsics is not set')
        return self._intrinsics
    
    def set_c2w(self, c2w=None) -> None:
        """
        Set the camera-to-world transformation matrix.

        Args:
            c2w (np.ndarray, optional): The camera-to-world transformation matrix. If not provided, it will be calculated as the inverse of w2c.

        Raises:
            ValueError: If w2c is not set and c2w is not provided.
            TypeError: If c2w is not a numpy array.
            ValueError: If c2w is not a 4x4 matrix.
        """
        if c2w is None:
            if self._w2c is None:
                raise ValueError('w2c must be set if c2w is not provided')
            self._c2w = np.linalg.inv(self.w2c)
        else:
            if not isinstance(c2w, np.ndarray):
                raise TypeError(f'Expected c2w as numpy array, got {type(c2w)}')
            if c2w.shape != (4, 4):
                raise ValueError(f'c2w must be a 4x4 matrix, got {c2w.shape}')
            self._c2w = c2w
            self._w2c = np.linalg.inv(c2w)
    
    @property
    def c2w(self) -> np.ndarray:
        """
        Get the camera-to-world transformation matrix.

        Returns:
            np.ndarray: The camera-to-world transformation matrix.

        Raises:
            AttributeError: If c2w is not set.
        """
        if self._c2w is None:
            raise AttributeError('c2w is not set')
        return self._c2w
