"""

Wire Harness — A2C training, one model per target configuration (stage).

Usage (from v0_MAPF/):
    python learn.py --stage 0

Train all 5 stages in parallel:
    for s in 0 1 2 3 4; do
        nohup python learn.py --stage $s \
            > logs/stage$s.out 2>&1 &
    done
    # NOTE: 5 runs × N_ENVS workers — drop N_ENVS in config.py to ~12
    # if the machine cannot take 160 env processes.

TensorBoard:
    tensorboard --logdir logs/a2c
"""

import argparse
import sys
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "disabled")

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def _make_env_fn(stage):
    def _init():
        import sys
        sys.path.insert(0, _ROOT)
        import wireharness_gym as _env_pkg  # noqa: F401
        import gymnasium as gym
        from stable_baselines3.common.monitor import Monitor
        return Monitor(gym.make("WireHarness-v0", stage=stage))
    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one A2C per target configuration.")
    parser.add_argument("--stage", type=int, required=True, choices=range(5),
                        help="Target configuration index (0-4) this model learns.")
    args = parser.parse_args()
    stage = args.stage

    import wireharness_gym  # noqa: F401  (registers WireHarness-v0)
    import gymnasium as gym
    from stable_baselines3 import A2C
    from stable_baselines3.common.callbacks import (
        BaseCallback, EvalCallback, CheckpointCallback,
    )
    from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

    from config import (
        TOTAL_STEPS, N_ENVS, EVAL_FREQ, N_EVAL_EPS, CKPT_FREQ,
        LOG_DIR, CKPT_DIR, POLICY_NET_ARCH,
        A2C_LR, A2C_N_STEPS, A2C_GAMMA, A2C_GAE_LAMBDA,
        A2C_ENT_COEF, A2C_VF_COEF, A2C_MAX_GRAD,
    )

    log_dir  = os.path.join(LOG_DIR, f"stage_{stage}")
    ckpt_dir = os.path.join(CKPT_DIR, f"stage_{stage}")

    print(f"Creating {N_ENVS} training envs (5 movers, stage {stage})...")
    vec_env = VecNormalize(
        SubprocVecEnv([_make_env_fn(stage)] * N_ENVS),
        norm_obs=True, norm_reward=False,
    )

    eval_env = VecNormalize(
        SubprocVecEnv([_make_env_fn(stage)]),
        norm_obs=True, norm_reward=False, training=False,
    )

    model = A2C(
        "MlpPolicy",
        vec_env,
        learning_rate=A2C_LR,
        n_steps=A2C_N_STEPS,
        gamma=A2C_GAMMA,
        gae_lambda=A2C_GAE_LAMBDA,
        ent_coef=A2C_ENT_COEF,
        vf_coef=A2C_VF_COEF,
        max_grad_norm=A2C_MAX_GRAD,
        use_rms_prop=True,
        policy_kwargs=dict(net_arch=POLICY_NET_ARCH),
        verbose=1,
        tensorboard_log=log_dir,
        device="cpu",
    )

    class _SaveVecNormOnBest(BaseCallback):
        """EvalCallback saves only the model; pair it with the obs_rms stats
        it was actually selected under (test-time normalization skew otherwise)."""
        def __init__(self, vecnorm, path):
            super().__init__()
            self._vecnorm = vecnorm
            self._path = path
            os.makedirs(os.path.dirname(path), exist_ok=True)
        def _on_step(self) -> bool:
            self._vecnorm.save(self._path)
            return True

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(ckpt_dir, "best"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPS,
        deterministic=True,
        verbose=1,
        callback_on_new_best=_SaveVecNormOnBest(
            vec_env, os.path.join(ckpt_dir, "best", f"stage{stage}_vecnorm.pkl")),
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(CKPT_FREQ // N_ENVS, 1),
        save_path=ckpt_dir,
        name_prefix=f"stage{stage}_a2c",
        verbose=1,
    )

    print(f"\nTraining A2C stage {stage} for {TOTAL_STEPS:,} steps "
          f"({N_ENVS} envs, obs=410, action=10)...")
    print(f"TensorBoard: tensorboard --logdir {log_dir}\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[eval_cb, ckpt_cb],
        progress_bar=False,
    )

    final_path = os.path.join(ckpt_dir, f"stage{stage}_final")
    norm_path  = os.path.join(ckpt_dir, f"stage{stage}_vecnorm.pkl")
    model.save(final_path)
    vec_env.save(norm_path)
    print(f"\nModel saved  → {final_path}.zip")
    print(f"Normalizer   → {norm_path}")

    print("\nRunning deterministic eval episode (random predecessor start)...")
    _plain_env = gym.make("WireHarness-v0", stage=stage)
    obs, _ = _plain_env.reset()
    # normalize with the trained running stats so the policy sees train-time obs
    obs = vec_env.normalize_obs(obs)
    total_reward = 0.0
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = _plain_env.step(action)
        obs = vec_env.normalize_obs(obs)
        total_reward += reward
        steps += 1

    n_goal  = info.get("n_at_goal", 0)
    outcome = "CONFIGURATION REACHED" if terminated else f"TIMEOUT ({n_goal}/5 at goal)"
    print(f"[{outcome}] steps={steps}  reward={total_reward:.2f}")

    _plain_env.close()
    eval_env.close()
    vec_env.close()
