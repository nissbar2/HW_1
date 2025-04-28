from jaxtyping import Float
from torch import Tensor
import torch


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""
    ones = torch.ones_like(points[..., :1])  # Shape: (*batch, 1)
    return torch.cat([points, ones], dim=-1)
    # raise NotImplementedError("This is your homework.")


def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""
    zeros = torch.zeros_like(points[..., :1])  # Shape: (*batch, 0)
    return torch.cat([points, zeros], dim=-1)


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""
    return torch.matmul(transform, xyz.unsqueeze(-1)).squeeze(-1)


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """
    world2cam = torch.linalg.inv(cam2world)  # shape: (*batch, 4, 4)
    return torch.matmul(world2cam, xyz.unsqueeze(-1)).squeeze(-1)  # shape: (*batch, 4)


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """
    return torch.matmul(cam2world, xyz.unsqueeze(-1)).squeeze(-1)


def project(
    xyz: Float[Tensor, "*#batch 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""
    x = xyz[..., 0] / xyz[..., 3]
    y = xyz[..., 1] / xyz[..., 3]
    z = xyz[..., 2] / xyz[..., 3]
    xyz = torch.stack([x, y, z], dim=-1)
    # Step 2: Apply intrinsics matrix
    uvw = torch.matmul(intrinsics, xyz.unsqueeze(-1)).squeeze(-1)  # shape: (*batch, 3)

    # Step 3: Divide by depth (perspective division)
    u = uvw[..., 0] / uvw[..., 2]
    v = uvw[..., 1] / uvw[..., 2]

    return torch.stack([u, v], dim=-1)  # shape: (*batch, 2)
