import os

import mujoco
import mujoco.viewer
from robot_descriptions import aloha_mj_description
from scipy.spatial.transform import Rotation


def find_body(parent, name):
    body = parent.first_body()
    while body:
        if body.name == name:
            return body
        else:
            # Recursively search child bodies.
            child_body = find_body(body, name)
            if child_body:
                return child_body
        body = parent.next_body(body)
    raise ValueError(f"Body '{name}' not found")


def aloha_with_marker():
    ###############################################################
    # Base aloha scene.
    ###############################################################

    spec = mujoco.MjSpec.from_file(aloha_mj_description.MJCF_PATH)

    ###############################################################
    # Add an aruco marker to the scene.
    # TODO: this should be another spec which lives elsewere,
    #       gets attached (using mujoco API to attach another spec)
    ###############################################################

    # Create a new texture for the tag.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec.add_texture(
        name="tag_texture",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=os.path.join(current_dir, "5x5_1000-187.png"),
    )
    spec.add_material(name="tag_material").textures[
        mujoco.mjtTextureRole.mjTEXROLE_RGB
    ] = "tag_texture"  # type: ignore

    # Mount a tag to the gripper.
    gripper_base = find_body(spec.worldbody, "right/gripper_base")
    tag_body = gripper_base.add_body(
        name="tag_187",
        pos=[0, -0.1, -0.032],
        quat=Rotation.from_euler("XYZ", [90, 180, 20], degrees=True).as_quat(),
    )
    # A small backing for the tag, so there's a nice visual boundary.
    tag_body.add_geom(
        name="tag_backing",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.06 / 2, 0.06 / 2, 0.0001],
    )
    # The actual tag.
    tag_body.add_geom(
        name="tag",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.05 / 2, 0.05 / 2, 0.001],  # Shape of the aruco marker.
        material="tag_material",
    )

    ###############################################################
    # Put a camera over the shoulder.
    ###############################################################

    # Global framebuffer stuff, for rendering. All off-screen
    # renderers can have a max resolution of this.
    getattr(spec.visual, "global").offheight = 720
    getattr(spec.visual, "global").offwidth = 1080

    # Add a camera to the worldbody, to simulate "over-the-shoulder".
    spec.worldbody.add_camera(
        name="tag_cam",
        pos=[0.4, -0.35, 0.8],
        quat=Rotation.from_euler("ZYX", [45, 0, 45], degrees=True).as_quat(
            scalar_first=True
        ),
        fovy=60,
        resolution=[1080, 720],
    )

    spec.compile()

    # Load in all the assets for mjcf. For some reason these don't resolve.
    def load_assets(asset_dir):
        """Loads all assets in a given directory as a dictionary of {name: byte data}"""
        assets = {}
        for root, _, files in os.walk(asset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                asset_key = os.path.relpath(file_path, asset_dir)  # Keep relative path
                with open(file_path, "rb") as f:
                    assets[asset_key] = f.read()
        return assets

    assets_dir = os.path.join(os.path.dirname(aloha_mj_description.MJCF_PATH), "assets")
    assets = load_assets(assets_dir)

    # Create a model from the spec.
    model = mujoco.MjModel.from_xml_string(spec.to_xml(), assets=assets)

    return model
