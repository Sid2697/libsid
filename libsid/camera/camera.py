
import numpy as np


class CameraManager():
    def __init__(self, camera_id):
        self._camera_id = camera_id
        self._w2c = None
        self._c2w = None
        self._intrinsics = None
        self.params = {
            'id': camera_id,
            'w2c': None,
            'c2w': None,
            'intrinsics': None,
        }
    
    @property
    def camera_id(self) -> str:
        """
        Get the camera id
        """
        return self._camera_id

    def setup(self, w2c: np.ndarray, intrinsics: np.ndarray) -> None:
        """
        One function to do all the setup
        """
        self.set_w2c(w2c)
        self.set_c2w()
        self.set_intrinsics(intrinsics)
    
    def set_w2c(self, w2c: np.ndarray) -> None:
        """
        Set the world to camera transformation matrix
        """
        if w2c.shape != (4, 4):
            raise ValueError(f'w2c must be a 4x4 matrix, got {w2c.shape}')
        if not isinstance(w2c, np.ndarray):
            raise TypeError(f'Expected w2c as numpy array, got {type(w2c)}')
        self._w2c = w2c
        self.params['w2c'] = w2c
    
    @property
    def w2c(self) -> np.ndarray:
        """
        Get the world to camera transformation matrix
        """
        if self._w2c is None:
            raise AttributeError('w2c is not set')
        return self._w2c

    def set_intrinsics(self, intrinsics: np.ndarray) -> None:
        """
        Set the camera intrinsics matrix
        """
        if intrinsics.shape != (3, 3):
            raise ValueError(f'intrinsics must be a 3x3 matrix, got {intrinsics.shape}')
        if not isinstance(intrinsics, np.ndarray):
            raise TypeError(f'Expected intrinsics as numpy array, got {type(intrinsics)}')
        self._intrinsics = intrinsics
        self.params['intrinsics'] = intrinsics
    
    @property
    def intrinsics(self) -> np.ndarray:
        """
        Get the camera intrinsics matrix
        """
        if self._intrinsics is None:
            raise AttributeError('intrinsics is not set')
        return self._intrinsics
    
    def set_c2w(self, c2w=None) -> None:
        """
        Set the camera to world transformation matrix
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
            self.params['w2c'] = self._w2c
        self.params['c2w'] = self._c2w
    
    @property
    def c2w(self) -> np.ndarray:
        """
        Get the camera to world transformation matrix
        """
        if self._c2w is None:
            raise AttributeError('c2w is not set')
        return self._c2w
    