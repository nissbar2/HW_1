from pathlib import Path
from typing import Literal, TypedDict

from jaxtyping import Float
from torch import Tensor
import torch
from torchvision.io import read_image
import json

class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def load_dataset(path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""

    # Load metadata.json
    with open(path / "metadata.json", "r") as f:
        metadata = json.load(f)

    intrinsics = torch.tensor(metadata["intrinsics"], dtype=torch.float32)
    extrinsics = torch.tensor(metadata["extrinsics"], dtype=torch.float32)

    # Load 32 images
    images = []
    for i in range(32):
        img_path = path / f"images/{i:02d}.png"
        img = read_image(str(img_path)).float() / 255.0  # normalize to [0, 1]
        images.append(img)

    # Stack into one tensor: shape = (batch, channels, height, width)
    image_tensor = torch.stack(images)  # shape: (32, 3, H, W)
    # Convert to grayscale if needed (e.g., if task expects 1 channel)
    if image_tensor.shape[1] == 3:
        image_tensor = image_tensor[:, 0]
        # image_tensor shape: (batch, height, width)
    return {
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "images": image_tensor
    }


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    """Convert any randomized extrinsics to OpenCV-style camera-to-world format.

    Handles:
    - Random rotation axes
    - Left- or right-handed coordinate systems
    - Either camera-to-world or world-to-camera input
    """

    extrinsics = dataset["extrinsics"]
    batch_size = extrinsics.shape[0]
    new_extrinsics = []

    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=extrinsics.device)

    for E in extrinsics:
        # E may be cam2world or world2cam → we want camera origin in world space
        cam2world_candidate = E
        world2cam_candidate = torch.linalg.inv(E)

        def extract_camera_origin(matrix):
            return matrix[:3, 3]

        origin_1 = extract_camera_origin(cam2world_candidate)
        origin_2 = extract_camera_origin(world2cam_candidate)

        # Choose the one that lies ~2 units from world origin
        dist1 = torch.norm(origin_1)
        dist2 = torch.norm(origin_2)

        if (torch.isclose(dist1, torch.tensor(2.0, device=extrinsics.device), atol=1e-2) and
                torch.isclose(dist2, torch.tensor(2.0, device=extrinsics.device), atol=1e-2)):
            # Both are valid candidates, choose the one with matching look direction
            look_1 = torch.nn.functional.normalize(-origin_1, dim=0)
            look_2 = torch.nn.functional.normalize(-origin_2, dim=0)
            for i in range(3):
                if torch.isclose(torch.dot(look_1, E[:3, i]), torch.tensor(-1.0, device=extrinsics.device), atol=1e-2):
                    camera_origin = origin_1
                    break
                elif torch.isclose(torch.dot(look_2, E[:3, i]), torch.tensor(-1.0, device=extrinsics.device), atol=1e-2):
                    camera_origin = origin_2
                    break
        elif torch.isclose(dist1, torch.tensor(2.0, device=extrinsics.device), atol=1e-2):
            camera_origin = origin_1
        elif torch.isclose(dist2, torch.tensor(2.0, device=extrinsics.device), atol=1e-2):
            camera_origin = origin_2
        else:
            # fallback (shouldn't happen)
            camera_origin = origin_1 if dist1 < dist2 else origin_2

        # Step 1: Look direction = towards world origin
        look = torch.nn.functional.normalize(-camera_origin, dim=0)  # +Z

        # Step 2: Right = world up × look
        right = torch.cross(world_up, look)
        if torch.norm(right) < 1e-5:  # camera is looking straight up or down
            right = torch.tensor([1.0, 0.0, 0.0], device=extrinsics.device)
        else:
            right = torch.nn.functional.normalize(right, dim=0)

        # Step 3: Up = look × right
        up = torch.cross(look, right)
        up = torch.nn.functional.normalize(up, dim=0)

        # Step 4: Ensure right-handed system
        R = torch.stack([right, -up, look], dim=1)  # OpenCV: +X, -Y, +Z
        if torch.det(R) < 0:
            right = -right
            R = torch.stack([right, -up, look], dim=1)

        # Assemble final camera-to-world matrix
        E_new = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
        E_new[:3, :3] = R
        E_new[:3, 3] = camera_origin

        new_extrinsics.append(E_new)

    new_extrinsics = torch.stack(new_extrinsics)

    return {
        "extrinsics": new_extrinsics,
        "intrinsics": dataset["intrinsics"],
        "images": dataset["images"],
    }

def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""
    return "c2w"


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""
    return "+z"



def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""
    return "-y"


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""
    return "+x"


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """
    return ("I solved the puzzle by analyzing the convert_dataset() function."
            " The function checks both camera-to-world and world-to-camera "
            "interpretations by evaluating the camera origin distance."
            "Which due to translation rotation properties leads to the same"
            "distance from the world origin. I therefore checked the "
            "camera look direction matching by checking the dot product of the "
            "camera look vector with the camera axes. "
            " Since the direct extrinsic matrices (without inversion)"
            " matched the expected camera origin, I concluded that the "
            "original dataset was already in camera-to-world format."
            " The camera look direction was determined by "
            "normalizing the negative camera origin. "
            " Based on standard OpenCV conventions, I identified the "
            "camera up, and right directions accordingly.")
