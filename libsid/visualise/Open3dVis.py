import os
import copy
import numpy as np
import open3d as o3d
from typing import List, Optional


class O3DVisualiser():
    def __init__(self):
        self.scene_objects = list()

    def add_obj(
            self,
            obj_path: str,
            obj_pose: np.ndarray,
            verts_scale: float = 1,
            color: Optional[List[float]] = None,
        ) -> o3d.geometry.TriangleMesh:
        """
        Adds an object to the scene by loading a 3D mesh from a file, transforming it, and applying color.

        Args:
            obj_path (str): The file path to the 3D object mesh.
            obj_pose (np.ndarray): A 4x4 transformation matrix to apply to the object.
            verts_scale (float, optional): A scaling factor for the vertices. Defaults to 1.
            color (list, optional): A list of three floats representing the RGB color to paint the object. Defaults to None.

        Returns:
            o3d.geometry.TriangleMesh: The transformed and colored 3D object mesh.
        """
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f'Object file not found: {obj_path}')
        obj_mesh = o3d.io.read_triangle_mesh(obj_path)
        obj_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(obj_mesh.vertices) / verts_scale)
        obj_mesh.compute_vertex_normals()
        if color is None:
            color = [0.5, 0.5, 0.5]
        obj_mesh.paint_uniform_color(color)
        if not isinstance(obj_pose, np.ndarray):
            raise ValueError(f'Expected obj_pose as numpy array, got {type(obj_pose)}')
        if obj_pose.shape != (4, 4):
            raise ValueError(f'Expected obj_pose as 4x4 matrix, got {obj_pose.shape}')
        obj_mesh = copy.deepcopy(obj_mesh).transform(obj_pose)
        self.scene_objects.append(obj_mesh)
        return obj_mesh

    def add_mesh(
            self,
            vertices: np.ndarray,
            faces: np.ndarray,
            mesh_pose: Optional[np.ndarray] = np.eye(4),
            apply_pose: bool = False,
            verts_scale: float = 1,
            color: Optional[List[float]] = None,
        ) -> o3d.geometry.TriangleMesh:
        """
        Adds a mesh to the scene.

        Args:
        -----------
        vertices : np.ndarray
            A numpy array of shape (N, 3) representing the vertices of the mesh.
        faces : np.ndarray
            A numpy array of shape (N, 3) representing the faces of the mesh.
        mesh_pose : np.ndarray, optional
            A 4x4 transformation matrix representing the pose of the mesh. Default is the identity matrix.
        apply_pose : bool, optional
            Whether to apply the mesh_pose transformation to the mesh. Default is False.
        verts_scale : float, optional
            A scaling factor for the vertices. Default is 1.
        color : Optional[List[float]], optional
            A list of three floats representing the RGB color of the mesh. Default is None, which sets the color to [0.5, 0.5, 0.5].

        Returns:
        --------
        o3d.geometry.TriangleMesh
            The created TriangleMesh object.

        Raises:
        -------
        ValueError
            If the vertices or faces are not numpy arrays or if their shapes are not (N, 3).
            If the mesh_pose is not a numpy array or if its shape is not (4, 4).
        """
        if not isinstance(vertices, np.ndarray):
            raise ValueError(f'Expected vertices as numpy array, got {type(vertices)}')
        if vertices.shape[1] != 3:
            raise ValueError(f'Vertices must be Nx3, got {vertices.shape}')
        if not isinstance(faces, np.ndarray):
            raise ValueError(f'Expected faces as numpy array, got {type(faces)}')
        if faces.shape[1] != 3:
            raise ValueError(f'Faces must be Nx3, got {faces.shape}')
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices) / verts_scale)
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces))
        mesh.compute_vertex_normals()
        if color is None:
            color = [0.5, 0.5, 0.5]
        mesh.paint_uniform_color(color)
        if apply_pose:
            if not isinstance(mesh_pose, np.ndarray):
                raise ValueError(f'Expected mesh_pose as numpy array, got {type(mesh_pose)}')
            if mesh_pose.shape != (4, 4):
                raise ValueError(f'Expected mesh_pose as 4x4 matrix, got {mesh_pose.shape}')
            mesh = copy.deepcopy(mesh).transform(mesh_pose)
        self.scene_objects.append(mesh)
        return mesh

    def add_frame(
            self,
            size: float = 1,
            pose: np.ndarray = np.eye(4),
        ) -> o3d.geometry.TriangleMesh:
        """
        Adds a coordinate frame to the scene.

        Args:
            size (float): The size of the coordinate frame. Default is 1.
            pose (np.ndarray): A 4x4 transformation matrix representing the pose of the frame. Default is the identity matrix.

        Returns:
            o3d.geometry.TriangleMesh: The coordinate frame as a TriangleMesh object.
        """
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        if not isinstance(pose, np.ndarray):
            raise ValueError(f'Expected pose as numpy array, got {type(pose)}')
        if pose.shape != (4, 4):
            raise ValueError(f'Expected pose as 4x4 matrix, got {pose.shape}')
        frame = copy.deepcopy(frame).transform(pose)
        self.scene_objects.append(frame)
        return frame
    
    def add_cam_frustum(
            self,
            intrinsic: np.ndarray,
            cam_pose: np.ndarray,
            image_size: tuple,
            scale: float = 1,
            color: Optional[List[float]] = None,
        ) -> o3d.geometry.LineSet:
        """
        Creates a 3D representation of a camera frustum using the given intrinsic parameters, camera pose, and image size.

        Args:
            intrinsic (np.ndarray): The camera intrinsic matrix (3x3).
            cam_pose (np.ndarray): The camera pose matrix (4x4).
            image_size (tuple): The size of the image (width, height).
            scale (float, optional): The scale factor for the frustum size. Default is 1.
            color (list, optional): The color of the frustum lines in RGB format. Default is [1, 0, 0] (red).

        Returns:
            o3d.geometry.LineSet: A LineSet object representing the camera frustum.

        Note:
            The far and near planes DO NOT reflect the actual camera frustum, but are scaled for visibility.
            This frustum is intended for visualization and debugging purposes using Open3D.
        """
        if not isinstance(intrinsic, np.ndarray):
            raise ValueError(f'Expected intrinsic as numpy array, got {type(intrinsic)}')
        if intrinsic.shape != (3, 3):
            raise ValueError(f'Expected intrinsic as 3x3 matrix, got {intrinsic.shape}')
        if not isinstance(cam_pose, np.ndarray):
            raise ValueError(f'Expected cam_pose as numpy array, got {type(cam_pose)}')
        if cam_pose.shape != (4, 4):
            raise ValueError(f'Expected cam_pose as 4x4 matrix, got {cam_pose.shape}')
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            raise ValueError(
                f'Expected image_size as a tuple (width, height), got {type(image_size)} with length {len(image_size)}'
            )

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        width, height = image_size
        
        # Compute frustum corners in camera coordinate space
        near = scale * 0.1  # Near plane (scaled for visibility)
        far = scale         # Far plane
        
        corners = np.array([
            [(0 - cx) * near / fx, (0 - cy) * near / fy, near],
            [(width - cx) * near / fx, (0 - cy) * near / fy, near],
            [(width - cx) * near / fx, (height - cy) * near / fy, near],
            [(0 - cx) * near / fx, (height - cy) * near / fy, near],
            [(0 - cx) * far / fx, (0 - cy) * far / fy, far],
            [(width - cx) * far / fx, (0 - cy) * far / fy, far],
            [(width - cx) * far / fx, (height - cy) * far / fy, far],
            [(0 - cx) * far / fx, (height - cy) * far / fy, far]
        ])
        
        # Transform frustum to world coordinates
        corners = (cam_pose[:3, :3] @ corners.T).T + cam_pose[:3, 3]
        
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Near plane edges
            [4, 5], [5, 6], [6, 7], [7, 4],  # Far plane edges
            [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
        ]
        
        if color is None:
            color = [1, 0, 0]
        colors = [color for _ in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        self.scene_objects.append(line_set)
        return line_set
    
    def render(self) -> None:
        """
        Visualises the scene with all added objects.
        """
        if not self.scene_objects:
            raise ValueError('No objects added to the scene')
        o3d.visualization.draw_geometries(self.scene_objects)
