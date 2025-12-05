import torch
import torch.nn.functional as F
import math
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    PointLights,
    look_at_view_transform,
)
from pytorch3d.utils import ico_sphere
from torchvision.utils import save_image

def create_textured_ico_sphere(level, device):
    mesh = ico_sphere(level=level, device=device)
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    
    verts_normalized = F.normalize(verts, p=2, dim=1)
    
    x, y, z = verts_normalized[:, 0], verts_normalized[:, 1], verts_normalized[:, 2]
    u = 0.5 + torch.atan2(z, x) / (2 * math.pi)
    v = 0.5 - torch.asin(torch.clamp(y, -1.0, 1.0)) / math.pi
    
    verts_uvs = torch.stack([u, v], dim=1)
    faces_uvs = faces
    
    # Grey texture
    texture_map = torch.ones(1, 512, 512, 3, device=device) * 0.5
    
    textures = TexturesUV(
        maps=texture_map,
        faces_uvs=[faces_uvs],
        verts_uvs=[verts_uvs],
    )
    
    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures,
    )
    
    # Explicitly set vertex normals
    mesh.verts_normals_packed = lambda: verts_normalized
    
    return mesh

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=None,
            raster_settings=RasterizationSettings(
                image_size=512,
                blur_radius=0.0,
                faces_per_pixel=1,
            ),
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=None,
            lights=PointLights(device=device, location=[[2.0, 2.0, 2.0]]),
        ),
    )
    
    mesh = create_textured_ico_sphere(level=4, device=device)
    
    R, T = look_at_view_transform(dist=2.5, elev=20, azim=45, device=device)
    camera = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    image = renderer(mesh, cameras=camera, lights=PointLights(device=device, location=[[2.0, 2.0, 2.0]]))
    
    save_image(image[..., :3].permute(0, 3, 1, 2), "debug_sphere.png")
    print("Saved debug_sphere.png")

if __name__ == "__main__":
    main()
