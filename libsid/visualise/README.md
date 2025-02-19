# Visualise

This module provides various classes to visualise 3D data using various libraries like, `matplotlib`, `open3d`, `pytorch3d`, etc.

# Open3dVis.py

This module provides a class `O3DVisualiser` for visualizing 3D objects, coordinate frames, and camera frustums using Open3D. The class allows adding various elements to a scene and visualizing them.

## Class `O3DVisualiser`

### Methods

#### `__init__(self)`

Initializes an instance of the `O3DVisualiser` class.

#### `add_obj(self, obj_path: str, obj_pose: np.ndarray, verts_scale: float = 1, color: list = None) -> o3d.geometry.TriangleMesh`

Adds an object to the scene by loading a 3D mesh from a file, transforming it, and applying color.

**Args:**
- `obj_path (str)`: The file path to the 3D object mesh.
- `obj_pose (np.ndarray)`: A 4x4 transformation matrix to apply to the object.
- `verts_scale (float, optional)`: A scaling factor for the vertices. Defaults to 1.
- `color (list, optional)`: A list of three floats representing the RGB color to paint the object. Defaults to None.

**Returns:**
- `o3d.geometry.TriangleMesh`: The transformed and colored 3D object mesh.

#### `add_frame(self, size: float = 1, pose: np.ndarray = np.eye(4)) -> o3d.geometry.TriangleMesh`

Adds a coordinate frame to the scene.

**Parameters:**
- `size (float)`: The size of the coordinate frame. Default is 1.
- `pose (np.ndarray)`: A 4x4 transformation matrix representing the pose of the frame. Default is the identity matrix.

**Returns:**
- `o3d.geometry.TriangleMesh`: The coordinate frame as a TriangleMesh object.

#### `add_cam_frustum(self, intrinsic: np.ndarray, cam_pose: np.ndarray, image_size: tuple, scale: float = 1, color: list = None) -> o3d.geometry.LineSet`

Creates a 3D representation of a camera frustum using the given intrinsic parameters, camera pose, and image size.

**Args:**
- `intrinsic (numpy.ndarray)`: The camera intrinsic matrix (3x3).
- `cam_pose (numpy.ndarray)`: The camera pose matrix (4x4).
- `image_size (tuple)`: The size of the image (width, height).
- `scale (float, optional)`: The scale factor for the frustum size. Default is 1.
- `color (list, optional)`: The color of the frustum lines in RGB format. Default is [1, 0, 0] (red).

**Returns:**
- `open3d.geometry.LineSet`: A LineSet object representing the camera frustum.

**Note:**
The far and near planes DO NOT reflect the actual camera frustum, but are scaled for visibility. This frustum is intended for visualization and debugging purposes using Open3D.

#### `visualise(self) -> None`

Visualizes the scene with all added objects.

**Raises:**
- `ValueError`: If no objects have been added to the scene.

# Pytorch3dVis.py

This module provides a class `Py3DVisualiser` for visualizing 3D objects using PyTorch3D. The class allows adding objects to a scene, transforming them, and rendering them with various camera parameters.

## Class `Py3DVisualiser`

### Methods

#### `__init__(self, img_height: int, img_width: int, device: str = 'cpu', projection: str = 'perspective')`

Initializes an instance of the `Py3DVisualiser` class with image dimensions, device, and projection type.

**Args:**
- `img_height (int)`: The height of the image.
- `img_width (int)`: The width of the image.
- `device (str)`: The device to use ('cpu' or 'cuda').
- `projection (str)`: The type of projection ('perspective' or 'orthographic').

#### `add_obj(self, mesh_path: str, verts_scale: float = 1)`

Adds an object to the scene from a mesh file.

**Args:**
- `mesh_path (str)`: The path to the mesh file.
- `verts_scale (float)`: The scale factor for the vertices.

**Raises:**
- `FileNotFoundError`: If the mesh file is not found.

#### `transform_obj(self, obj_R: torch.tensor, obj_T: torch.tensor)`

Transforms the object in the scene using rotation and translation matrices.

**Args:**
- `obj_R (torch.tensor)`: The 3x3 rotation matrix.
- `obj_T (torch.tensor)`: The 3x1 translation vector.

**Raises:**
- `NotImplementedError`: If there is more than one object in the scene.

#### `get_mesh_renderer(self, lights: Optional[py3drend.PointLights] = None) -> py3drend.MeshRenderer`

Returns a PyTorch3D Mesh Renderer.

**Args:**
- `lights (Optional[py3drend.PointLights])`: A default PyTorch3D lights object.

**Returns:**
- `py3drend.MeshRenderer`: The mesh renderer.

#### `get_textured_mesh(self, mesh: py3d.structures.Meshes) -> py3d.structures.Meshes`

Applies a simple texture to the mesh.

**Args:**
- `mesh (py3d.structures.Meshes)`: The mesh to be textured.

**Returns:**
- `py3d.structures.Meshes`: The textured mesh.

#### `get_extened_intrinsics(self, intrinsic: torch.tensor) -> torch.tensor`

Extends the intrinsic matrix to a 4x4 matrix for perspective projection.

**Args:**
- `intrinsic (torch.tensor)`: The 3x3 intrinsic matrix.

**Returns:**
- `torch.tensor`: The extended 4x4 intrinsic matrix.

**Raises:**
- `NotImplementedError`: If the projection type is not 'perspective'.

#### `render_object(self, cam_R: torch.tensor, cam_T: torch.tensor, intrinsic: torch.tensor)`

Renders the object in the scene using the camera parameters.

**Args:**
- `cam_R (torch.tensor)`: The 3x3 camera rotation matrix.
- `cam_T (torch.tensor)`: The 3x1 camera translation vector.
- `intrinsic (torch.tensor)`: The 3x3 intrinsic matrix.

**Raises:**
- `NotImplementedError`: If there is more than one object in the scene.