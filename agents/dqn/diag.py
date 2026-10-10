from Environment import Traffic_Env
import numpy as np

env = Traffic_Env()

# ── basic checks ──────────────────────────────────────────────────────
obs, info = env.reset(seed=42)
print("=== RESET ===")
print(f"Obs:       {obs}")
print(f"Obs shape: {obs.shape}  (expected (10,))")
print(f"Obs dtype: {obs.dtype}  (expected int)")
print(f"Obs valid: {env.observation_space.contains(obs)}  (expected True)")
print()

# ── 20 random steps ───────────────────────────────────────────────────
print("=== STEPS ===")
phases_seen = set()

for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    phases_seen.add(action)

    print(f"Step {step+1:2d} | action={action} ({info['phase']:<4}) | reward={reward:8.2f} | delta_Q={info['delta_Q']:4d}")
    print(f"         queues (str/left): N={obs[0]}/{obs[1]}  S={obs[2]}/{obs[3]}  E={obs[4]}/{obs[5]}  W={obs[6]}/{obs[7]}")
    print(f"         phase={obs[8]}  amb_required={obs[9]}  amb_present={info['ambulance_present']}")
    print(f"         starv_counts={info['starvation_counts']}  starv_penalty={info['starvation_penalty']}")
    print(f"         amb_penalty={info['ambulance_penalty']}")
    print(f"         obs_valid={env.observation_space.contains(obs)}")
    print()

    if terminated or truncated:
        print("Episode ended early — resetting")
        obs, info = env.reset()

print(f"Phases seen in 20 steps: {phases_seen}  (ideally all of {{0,1,2,3}})")

# ── ambulance persistence check ───────────────────────────────────────
print()
print("=== AMBULANCE PERSISTENCE TEST ===")
obs, _ = env.reset(seed=7)

# force an ambulance into state
env.state_space[9] = 2  # requires EWA
obs = env.state_space.copy()
print(f"Forced ambulance requiring phase 2 (EWA). state[9]={env.state_space[9]}")

for step in range(6):
    # deliberately pick wrong action to test persistence
    action = 0  # NSA — wrong phase
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step {step+1}: action=0 (NSA) | amb_required={obs[9]} | amb_present={info['ambulance_present']} | amb_penalty={info['ambulance_penalty']}")
    if obs[9] == 4:
        print("  !! Ambulance cleared without being served — persistence broken")
        break
    if obs[9] == 2:
        print("  ✓ Ambulance persisting correctly")

# now serve it
action = 2  # EWA — correct phase
obs, reward, terminated, truncated, info = env.step(action)
print(f"  Served with action=2 (EWA) | amb_required={obs[9]} | amb_present={info['ambulance_present']}")
if obs[9] == 4:
    print("  ✓ Ambulance cleared correctly after being served")
else:
    print("  !! Ambulance did not clear after correct phase served")

# ── observation space final check ─────────────────────────────────────
print()
print("=== FINAL OBS SPACE CHECK ===")
print(f"observation_space: {env.observation_space.nvec}")
print(f"Expected:          [100 100 100 100 100 100 100 100   4   5]")