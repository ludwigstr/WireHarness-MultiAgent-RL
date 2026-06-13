"""
V0.5 Wire Harness — chained evaluation & video recording (5 movers).

Runs the per-stage A2C models back to back in ONE episode: model k drives all
movers until configuration k is reached (all movers inside the goal radius
simultaneously — the env's terminated signal), then the next model in --order
takes over via env.set_stage().

Usage:
    python test.py                       # Konf 1→2→3→4→5
    python test.py --order 4 2 3 1 5     # arbitrary order
    python test.py --order 3             # single stage
    python test.py --start random        # random predecessor start

--order takes 1-indexed configuration numbers (Konf 1..5 = stage 0..4).
Each stage's observations are normalized with that stage's own VecNormalize
statistics (stage{k}_vecnorm.pkl) before being fed to its model.
"""

import sys
import os
import argparse

os.environ.setdefault("MUJOCO_GL", "egl")

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def _resolve_model_paths(ckpt_dir, stage):
    """Prefer the EvalCallback best model (paired with the vecnorm stats saved
    at the same moment), fall back to the final save. Returns (model, vecnorm)."""
    stage_dir = os.path.join(ckpt_dir, f"stage_{stage}")
    final_vn  = os.path.join(stage_dir, f"stage{stage}_vecnorm.pkl")
    candidates = [
        (os.path.join(stage_dir, "best", "best_model.zip"),
         os.path.join(stage_dir, "best", f"stage{stage}_vecnorm.pkl")),
        (os.path.join(stage_dir, f"stage{stage}_final.zip"), final_vn),
    ]
    for model_path, vn_path in candidates:
        if os.path.exists(model_path):
            if not os.path.exists(vn_path):
                vn_path = final_vn if os.path.exists(final_vn) else None
            return model_path, vn_path
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Chained per-stage evaluation.")
    parser.add_argument("--order", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="1-indexed configuration order, e.g. --order 4 2 3 1 5")
    parser.add_argument("--ckpt-dir", default=os.path.join(_ROOT, "checkpoints", "a2c"))
    parser.add_argument("--video",    default=os.path.join(_ROOT, "videos", "chained_eval.mp4"))
    parser.add_argument("--start",    choices=["initial", "random"], default="initial",
                        help="Start layout: XML rest positions or a random predecessor config.")
    args = parser.parse_args()

    if any(not 1 <= k <= 5 for k in args.order):
        print(f"[ERROR] --order entries must be in 1..5, got {args.order}")
        sys.exit(1)
    stage_order = [k - 1 for k in args.order]

    import wireharness_gym  # noqa: F401  (registers WireHarness-v0)
    import gymnasium as gym
    from stable_baselines3 import A2C
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

    # ── Load per-stage models + their VecNormalize stats ──────────────────
    models, vecnorms = {}, {}
    norm_env = None
    for stage in sorted(set(stage_order)):
        model_path, vn_path = _resolve_model_paths(args.ckpt_dir, stage)
        if model_path is None:
            print(f"[ERROR] No model for stage {stage} under "
                  f"{os.path.join(args.ckpt_dir, f'stage_{stage}')}")
            print(f"Train it with: python learn.py --stage {stage}")
            sys.exit(1)
        print(f"Loading stage {stage} model: {model_path}")
        models[stage] = A2C.load(model_path)

        if vn_path is not None:
            if norm_env is None:   # spaces-only venv, shared by all loads
                norm_env = DummyVecEnv([lambda: gym.make("WireHarness-v0")])
            vn = VecNormalize.load(vn_path, norm_env)
            vn.training = False
            vn.norm_reward = False
            vecnorms[stage] = vn
        else:
            print(f"[WARN] No VecNormalize stats for stage {stage} — "
                  f"using raw observations.")

    # ── One chained episode ───────────────────────────────────────────────
    # Step the unwrapped env directly: gym wrappers refuse step() after
    # terminated=True, but here each terminated marks a stage hand-over.
    raw_env = gym.make("WireHarness-v0", stage=stage_order[0]).unwrapped
    obs, _ = raw_env.reset(options={"start": args.start})
    raw_env.start_video(args.video)

    chain_reward = 0.0
    chain_steps  = 0
    results      = []
    aborted      = False

    for pos, stage in enumerate(stage_order):
        raw_env.set_stage(stage)
        raw_env.sim_step = 0          # fresh 60 s budget for every stage
        obs = raw_env._get_obs()      # refresh target features for the new stage

        stage_reward = 0.0
        stage_steps  = 0
        terminated = truncated = False

        while not (terminated or truncated):
            if stage in vecnorms:
                model_obs = vecnorms[stage].normalize_obs(obs)
            else:
                model_obs = obs
            action, _ = models[stage].predict(model_obs, deterministic=True)
            obs, reward, terminated, truncated, info = raw_env.step(action)
            stage_reward += reward
            stage_steps  += 1

        chain_reward += stage_reward
        chain_steps  += stage_steps
        # terminated alone is NOT success: the NaN-action guard and the physics
        # airbag also terminate. A stage counts as reached only with all 5
        # movers actually at goal and no failure marker.
        reached = (terminated and info.get("n_at_goal", 0) == 5
                   and not info.get("physics_unstable")
                   and not info.get("nan_action"))
        if reached:
            outcome = "REACHED"
        elif terminated:
            outcome = ("PHYSICS UNSTABLE" if info.get("physics_unstable")
                       else "NAN ACTION" if info.get("nan_action")
                       else f"FAILED ({info.get('n_at_goal', 0)}/5 at goal)")
        else:
            outcome = f"TIMEOUT ({info.get('n_at_goal', 0)}/5 at goal)"
        results.append((args.order[pos], outcome, stage_steps, stage_reward))
        print(f"[Konf {args.order[pos]}] [{outcome}] steps={stage_steps}  "
              f"reward={stage_reward:.2f}")

        if not reached:
            aborted = True
            print("[ABORT] Stage not completed — stopping the chain here.")
            break

    raw_env.finish_video()
    raw_env.close()
    if norm_env is not None:
        norm_env.close()

    print("\n──────── Chain summary ────────")
    for konf, outcome, steps, rew in results:
        print(f"  Konf {konf}: {outcome:<28} steps={steps:<6} reward={rew:.2f}")
    done_n = sum(1 for _, o, _, _ in results if o == "REACHED")
    status = "CHAIN COMPLETE" if not aborted else "CHAIN INCOMPLETE"
    print(f"[{status}] {done_n}/{len(stage_order)} configurations  "
          f"total steps={chain_steps}  total reward={chain_reward:.2f}")


if __name__ == "__main__":
    main()
