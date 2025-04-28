import torch
from jaxtyping import Float
from torch import Tensor
import matplotlib.pyplot as plt

from src.geometry import *


def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """
    batch_size = extrinsics.shape[0]
    height, width = resolution

    # Step 1: Homogenize the 3D vertices (shape: [vertex, 4])
    vertices_h = homogenize_points(vertices)  # (vertex, 4)

    # Step 2: Expand to shape (batch, vertex, 4) for batched processing
    vertices_h = vertices_h.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, vertex, 4)
    vertices_h = vertices_h.permute(1, 0, 2)  # from (batch, V, 4) to (V, batch, 4)
    # Step 3: Transform to camera space
    cam_coords = transform_world2cam(vertices_h, extrinsics)  # (vertex, batch, 4)
    z = cam_coords[..., 2]
    in_front = z > 0
    print(f"Number of points in front of camera: {in_front.sum()}")
    # Step 4: Project to 2D pixel coordinates
    pixel_coords = project(cam_coords, intrinsics)  # (batch, vertex, 2)
    pixel_coords[..., 0] *= width
    pixel_coords[..., 1] *= height
    print("Pixel coords stats:", torch.isnan(pixel_coords).any(), torch.isinf(pixel_coords).any())
    print("Pixel coord range:", pixel_coords.min(), pixel_coords.max())
    # Step 5: Create white canvas
    canvas = torch.ones(batch_size, height, width, dtype=torch.float32, device=vertices.device)
    # Step 6: Rasterize points onto the canvas
    pixel_coords = pixel_coords.permute(1, 0, 2)
    for b in range(batch_size):
        uvs = pixel_coords[b]  # (vertex, 2)

        # Round and clamp to image bounds
        x = uvs[:, 0].long()
        y = uvs[:, 1].long()
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        print(f"Batch {b}: {valid.sum()} points are valid")
        x = x[valid]
        y = y[valid]

        before = canvas[b].sum()
        canvas[b, y, x] = 0.0
        after = canvas[b].sum()
        print(f"Canvas changed: {before != after}")
    return canvas  # (batch, height, width)




