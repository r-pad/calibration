from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm


def se3_loss(T1, T2, alpha=1000.0):
    """
    Computes the loss between two rigid transformations T1 and T2.

    Args:
        T1: (4,4) jnp.array, homogeneous transformation matrix (R1 | t1)
        T2: (4,4) jnp.array, homogeneous transformation matrix (R2 | t2)
        alpha: Weighting factor between rotation and translation loss.

    Returns:
        Scalar loss value.
    """
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]

    # Rotation loss using Frobenius norm of log(R1^T R2)
    rotation_loss = ((R1.T @ R2 - jnp.eye(3)) ** 2).sum()

    # Translation loss (L2 loss)
    translation_loss = ((t1 - t2) ** 2).sum()

    return rotation_loss + alpha * translation_loss


def jointly_optimize_gripper_cam(
    Ts_world_from_gripper_base,
    Ts_camera_from_tag,
    lr=0.001,
    nsteps=10000,
    T_gripper_base_from_tag_init=None,
    alpha=10.0,
    pretrain_thresh=5000,
) -> Tuple[jax.Array, jax.Array]:
    # Check shapes.
    chex.assert_shape(Ts_world_from_gripper_base, (None, 4, 4))
    chex.assert_shape(Ts_camera_from_tag, (None, 4, 4))
    chex.assert_equal_shape([Ts_world_from_gripper_base, Ts_camera_from_tag])
    if T_gripper_base_from_tag_init is not None:
        chex.assert_shape(T_gripper_base_from_tag_init, (4, 4))

    # Turn on jax nan check
    jax.config.update("jax_debug_nans", True)

    # Initialize parameters (one for T_gripper_base_from_tag and one for
    # T_world_from_camera).
    one_transform_params = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    params = jnp.concatenate([one_transform_params, one_transform_params])

    # Explicitly initialize parameters if we have a good a priori
    # estimate.
    if T_gripper_base_from_tag_init is not None:
        # Form our representation.
        init_params = jnp.concatenate([
            T_gripper_base_from_tag_init[:3, 0],
            T_gripper_base_from_tag_init[:3, 1],
            T_gripper_base_from_tag_init[:3, 3],
        ])
        params = jnp.concatenate([init_params, params[9:]])

    # Map a single transform's params to a transformation matrix,
    # using the Gram-Schmidt process to ensure orthogonality.
    # This is from "On the Continuity of Rotation Representations in Neural Networks"
    # https://arxiv.org/abs/1812.07035
    def params_to_T(params):
        ex_u, ey_u = params[:3], params[3:6]
        ez_u = jnp.cross(ex_u, ey_u)
        ez = ez_u / (jnp.linalg.norm(ez_u) + 1e-6)
        ex_u = jnp.cross(ey_u, ez)
        ex = ex_u / (jnp.linalg.norm(ex_u) + 1e-6)
        ey = ey_u / (jnp.linalg.norm(ey_u) + 1e-6)
        R = jnp.stack([ex, ey, ez], axis=1)
        t = params[6:]
        A = jnp.eye(4)
        A = A.at[:3, :3].set(R)
        A = A.at[:3, 3].set(t)
        return A

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params):
        # Extract transformation matrices.
        T_gripper_base_from_tag = params_to_T(params[:9])
        T_world_from_camera = params_to_T(params[9:])

        # Get the LHS and RHS transformation matrices, which should be equal.
        lhs = jnp.matmul(Ts_world_from_gripper_base, T_gripper_base_from_tag)
        rhs = jnp.matmul(T_world_from_camera, Ts_camera_from_tag)

        return jax.vmap(lambda a, b: se3_loss(a, b, alpha=alpha))(lhs, rhs).mean()

    with tqdm(range(nsteps)) as pbar:
        for it in pbar:
            loss, grads = jax.value_and_grad(loss_fn)(params)

            if it < pretrain_thresh and T_gripper_base_from_tag_init is not None:
                grads = jax.tree_map(
                    lambda g: jnp.where(jnp.arange(len(g)) < 9, 0, g), grads
                )

            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            pbar.set_description(f"Loss: {loss:.4f}")

            if it % 1000 == 0 and False:
                print(f"Optimized T_gripper_base_from_tag:\n{params_to_T(params[:9])}")
                print(f"Optimized T_world_from_camera:\n{params_to_T(params[9:])}")

    T_gripper_base_from_tag = params_to_T(params[:9])  # type: ignore
    T_world_from_camera = params_to_T(params[9:])  # type: ignore
    return T_gripper_base_from_tag, T_world_from_camera
