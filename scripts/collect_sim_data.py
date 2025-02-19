import pickle
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import numpy as np
import typer
from scipy.spatial.transform import Rotation

from rpad.calibration.aloha_with_marker import aloha_with_marker


def collect(n_samples: int = 20, out_dir: Path = Path("./captures")):
    data_dict = {}

    # Create a time-based output directory
    output_dir = out_dir / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the aloha model + data.
    model = aloha_with_marker()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # Forward once to initialize.

    # Use the latest Mujoco Renderer API
    width, height = 1080, 720
    renderer = mujoco.Renderer(model, height, width)
    renderer.update_scene(data, camera=model.cam("tag_cam").id)

    frustum_near = renderer.scene.camera[0].frustum_near
    frustum_top = renderer.scene.camera[0].frustum_top
    f_y = (height / 2) * (frustum_near / frustum_top)

    # Extract field of view (in degrees)
    # fovy = model.cam("tag_cam").fovy.item()

    # Convert fovy to focal length (assuming square pixels)
    # f_y = (height / 2) / np.tan(np.deg2rad(fovy / 2))
    f_x = f_y  # Assuming square pixels

    # Compute principal point (image center)
    c_x = width / 2
    c_y = height / 2

    # Intrinsics matrix
    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

    # Save the intrinsics matrix to a file
    np.savetxt(str(output_dir / "intrinsics.txt"), K)

    # Get the arm out of the way.
    data.joint("left/shoulder").qpos = -1.5
    data.actuator("left/shoulder").ctrl = -1.5
    data.joint("left/elbow").qpos = 1.5
    data.actuator("left/elbow").ctrl = 1.5

    counter = 0
    ready = False

    def set_ready(keycode):
        nonlocal ready
        ready = True

    # Interactively collect data.
    with mujoco.viewer.launch_passive(model, data, key_callback=set_ready) as viewer:  # type: ignore
        while viewer.is_running():
            if counter >= n_samples:
                break
            if ready:
                # Render the scene from the camera
                renderer.update_scene(data, camera=model.cam("tag_cam").id)
                rgb_image = renderer.render()
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                img_name = f"image_{counter:04d}.png"
                cv2.imwrite(str(output_dir / img_name), bgr_image)

                def get_world_from_frame(entity):
                    if hasattr(entity, "xpos"):
                        entity_pos = entity.xpos
                        entity_quat = entity.xquat
                    else:
                        entity_pos = entity.pos
                        entity_quat = entity.quat

                    T_world_from_entity = np.eye(4)
                    T_world_from_entity[:3, :3] = Rotation.from_quat(
                        entity_quat, scalar_first=True
                    ).as_matrix()
                    T_world_from_entity[:3, 3] = entity_pos
                    return T_world_from_entity

                # Save these poses in the dictionary.
                data_dict[img_name] = {
                    "T_world_from_camera": get_world_from_frame(model.cam("tag_cam")),
                    "T_world_from_base_link": get_world_from_frame(
                        data.body("right/base_link")
                    ),
                    "T_world_from_gripper_base": get_world_from_frame(
                        data.body("right/gripper_base")
                    ),
                    "T_world_from_tag": get_world_from_frame(data.body("tag_187")),
                }

                counter += 1
                ready = False
            mujoco.mj_step(model, data)
            viewer.sync()

    # Dump w/ pickle
    with open(str(output_dir / "poses.pkl"), "wb") as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    typer.run(collect)
