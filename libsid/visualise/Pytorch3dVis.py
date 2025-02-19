import os
import torch
import pytorch3d as py3d
from typing import Optional
import matplotlib.pyplot as plt
from pytorch3d import io as py3dio
from pytorch3d import renderer as py3drend


class Py3DVisualiser():
    def __init__(
            self,
            img_height: int,
            img_width: int,
            device: str = 'cpu',
            projection: str = 'perspective',
    ):
        """
        Initialize the Py3DVisualiser with image dimensions, device, and projection type.

        Args:
            img_height (int): The height of the image.
            img_width (int): The width of the image.
            device (str): The device to use ('cpu' or 'cuda').
            projection (str): The type of projection ('perspective' or 'orthographic').
        """
        self.img_height = img_height
        self.img_width = img_width
        self.device = device
        self.projection = projection
        self.scene_objects = list()
    
    def add_obj(
            self,
            mesh_path: str,
            verts_scale: float = 1,
        ):
        """
        Add an object to the scene from a mesh file.

        Args:
            mesh_path (str): The path to the mesh file.
            verts_scale (float): The scale factor for the vertices.

        Raises:
            FileNotFoundError: If the mesh file is not found.
        """
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f'Mesh file not found: {mesh_path}')
        mesh = py3dio.load_objs_as_meshes([mesh_path], device=self.device)
        mesh.verts_list()[0] = mesh.verts_list()[0] / verts_scale
        self.scene_objects.append(mesh)

    def transform_obj(
            self,
            obj_R: torch.tensor,
            obj_T: torch.tensor,
        ):
        """
        Transform the object in the scene using rotation and translation matrices.

        Args:
            obj_R (torch.tensor): The 3x3 rotation matrix.
            obj_T (torch.tensor): The 3x1 translation vector.

        Raises:
            NotImplementedError: If there is more than one object in the scene.
        """
        if len(self.scene_objects) > 1:
            raise NotImplementedError('Current implementation assumes one object in the scene')
        obj_R = obj_R.to(torch.float32).to(self.device)
        obj_T = obj_T.to(torch.float32).to(self.device)
        mesh = self.scene_objects[0]
        obj_verts_transformed = torch.matmul(
            mesh.verts_list()[0].to(torch.float32),
            obj_R.T,
        ) + obj_T
        obj_mesh = py3d.structures.Meshes(
            verts=[obj_verts_transformed],
            faces=[mesh.faces_list()[0]],
        )
        self.scene_objects[0] = obj_mesh

    def get_mesh_renderer(
                self, lights: Optional[py3drend.PointLights] = None
        ) -> py3drend.MeshRenderer:
        """
        Returns a Pytorch3D Mesh Renderer.

        Args:
            lights (Optional[py3drend.PointLights]): A default Pytorch3D lights object.

        Returns:
            py3drend.MeshRenderer: The mesh renderer.
        """
        raster_settings = py3drend.RasterizationSettings(
                image_size=(self.img_height, self.img_width),
                blur_radius=0.0,
                faces_per_pixel=1,
            )
        renderer = py3drend.MeshRenderer(
            rasterizer=py3drend.MeshRasterizer(raster_settings=raster_settings),
            shader=py3drend.HardPhongShader(device=self.device, lights=lights),
        )
        return renderer

    def get_textured_mesh(
            self,
            mesh: py3d.structures.Meshes,
        ) -> py3d.structures.Meshes:
        """
        Apply a simple texture to the mesh.

        Args:
            mesh (py3d.structures.Meshes): The mesh to be textured.

        Returns:
            py3d.structures.Meshes: The textured mesh.
        """
        vertices = mesh.verts_list()[0].to(torch.float32).unsqueeze(0).to(self.device)
        faces = mesh.faces_list()[0].to(torch.float32).unsqueeze(0).to(self.device)
        textures = torch.ones_like(vertices, dtype=torch.float32).to(self.device)
        textures = textures * torch.tensor([1, 1, 0.5]).to(self.device)
        mesh_textured = py3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=py3d.renderer.TexturesVertex(textures),
        )
        return mesh_textured

    def get_extened_intrinsics(
            self,
            intrinsic: torch.tensor,
        ) -> torch.tensor:
        """
        Extend the intrinsic matrix to a 4x4 matrix for perspective projection.

        Args:
            intrinsic (torch.tensor): The 3x3 intrinsic matrix.

        Returns:
            torch.tensor: The extended 4x4 intrinsic matrix.

        Raises:
            NotImplementedError: If the projection type is not 'perspective'.
        """
        extended_K = torch.zeros(4, 4).to(self.device)
        if self.projection == 'perspective':
            extended_K[:3, :3] = intrinsic.to(torch.float32).to(self.device)
            extended_K[2, 3] = 1
            extended_K[3, 2] = 1
            extended_K[2, 2] = 0
        else:
            raise NotImplementedError(f'Projection {self.projection} not implemented')
        return extended_K

    def render_object(
            self, 
            cam_R: torch.tensor,
            cam_T: torch.tensor,
            intrinsic: torch.tensor,
        ):
        """
        Render the object in the scene using the camera parameters.

        Args:
            cam_R (torch.tensor): The 3x3 camera rotation matrix.
            cam_T (torch.tensor): The 3x1 camera translation vector.
            intrinsic (torch.tensor): The 3x3 intrinsic matrix.

        Raises:
            NotImplementedError: If there is more than one object in the scene.
        """
        if len(self.scene_objects) > 1:
            raise NotImplementedError('Current implementation assumes one object in the scene')
        mesh = self.scene_objects[0]
        mesh_textured = self.get_textured_mesh(mesh)
        lights = py3drend.PointLights(location=[[0, 0, -1]]).to(self.device)
        renderer = self.get_mesh_renderer(lights=lights)
        extended_K = self.get_extened_intrinsics(intrinsic)

        cameras = py3drend.PerspectiveCameras(
            R=cam_R.view(1, 3, 3).to(torch.float32).to(self.device),
            T=cam_T.view(1, 3).to(torch.float32).to(self.device),
            K=extended_K.view(1, 4, 4).to(torch.float32).to(self.device),
            in_ndc=False,
            image_size = [[self.img_height, self.img_width]],
        ).to(self.device)
        rend = renderer(mesh_textured, cameras=cameras, lights=lights)
        plt.imshow(rend.numpy()[0, ..., :3], alpha=0.4)
        plt.show()
