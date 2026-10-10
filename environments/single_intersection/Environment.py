import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Traffic_Env(gym.Env):
    """
    Adaptive traffic signal control environment.

    State (10 values):
        [N_str, N_left, S_str, S_left, E_str, E_left, W_str, W_left,
         current_phase, ambulance_required_phase]

    Actions (4):
        0: NSA  — clears N_straight, S_straight
        1: NSLA — clears N_left,     S_left
        2: EWA  — clears E_straight, W_straight
        3: EWLA — clears E_left,     W_left

    Ambulance encoding:
        state[9] = 0–3: required phase
        state[9] = 4:   no ambulance
    """

    def __init__(self):
        super().__init__()

        # observation space: 8 queues (0–99) + phase (0–3) + ambulance (0–4)
        self.observation_space = spaces.MultiDiscrete(
            [100, 100, 100, 100, 100, 100, 100, 100, 4, 5]
        )
        self.action_space = spaces.Discrete(4)

        # internal state — initialised properly in reset()
        self.state_space = np.zeros(10, dtype=int)

        self.action_to_phase = {
            0: "NSA",
            1: "NSLA",
            2: "EWA",
            3: "EWLA",
        }

        # which state indices each phase clears
        self.phase_serves = {
            0: [0, 2],   # NSA:  N_straight, S_straight
            1: [1, 3],   # NSLA: N_left,     S_left
            2: [4, 6],   # EWA:  E_straight, W_straight
            3: [5, 7],   # EWLA: E_left,     W_left
        }

        # 70/30 straight/left split on base rates [2, 2, 4, 4]
        # order matches state indices 0–7
        self.arrival_rates = [1.4, 0.6, 1.4, 0.6, 2.8, 1.2, 2.8, 1.2]

        self.saturation_flow = 5      # vehicles cleared per served queue per step
        self.starvation_threshold = 10
        self.max_step = 200

        # reward weights
        self.lambda_1 = 2.0   # ambulance non-compliance weight
        self.lambda_2 = 0.3   # starvation weight

        # penalty magnitudes
        self.ambulance_penalty_value = 100
        self.starvation_penalty_per_phase = 30

    # ------------------------------------------------------------------ #
    #  Generators                                                          #
    # ------------------------------------------------------------------ #

    def ambulance_phase_generator(self):
        """15% chance of ambulance on a random required phase."""
        if np.random.randint(1, 101) >= 85:
            return np.random.randint(0, 4)
        return 4

    # ------------------------------------------------------------------ #
    #  Reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.phase_starvation_counter = [0, 0, 0, 0]

        # randomise initial queues — low values so agent starts from a
        # manageable state, not already saturated
        self.state_space = np.array([
            np.random.randint(0, 20),  # N_straight
            np.random.randint(0, 10),  # N_left
            np.random.randint(0, 20),  # S_straight
            np.random.randint(0, 10),  # S_left
            np.random.randint(0, 20),  # E_straight
            np.random.randint(0, 10),  # E_left
            np.random.randint(0, 20),  # W_straight
            np.random.randint(0, 10),  # W_left
            0,                          # current phase — start at NSA
            4,                          # no ambulance
        ], dtype=int)

        return self.state_space.copy(), {}

    # ------------------------------------------------------------------ #
    #  Step                                                                #
    # ------------------------------------------------------------------ #

    def step(self, action):
        self.current_step += 1

        # ── 1. store previous queues for delta_Q ──────────────────────
        prev_queues = self.state_space[0:8].copy()

        # ── 2. arrivals — all queues receive vehicles every step ───────
        arrivals = np.random.poisson(self.arrival_rates)

        # ── 3. departures — only served queues clear ───────────────────
        departures = np.zeros(8, dtype=int)
        for idx in self.phase_serves[action]:
            departures[idx] = min(self.state_space[idx], self.saturation_flow)

        # ── 4. update queues ───────────────────────────────────────────
        new_queues = self.state_space[0:8] + arrivals - departures
        self.state_space[0:8] = np.clip(new_queues, 0, 99)

        # ── 5. update current phase in state ───────────────────────────
        self.state_space[8] = action

        # ── 6. starvation counters — per phase ─────────────────────────
        for i in range(4):
            if i == action:
                self.phase_starvation_counter[i] = 0
            else:
                self.phase_starvation_counter[i] += 1

        starvation_penalty = 0
        for count in self.phase_starvation_counter:
            if count > self.starvation_threshold:
                starvation_penalty += self.starvation_penalty_per_phase

        # ── 7. ambulance compliance — check BEFORE updating state[9] ───
        amb_required = self.state_space[9]
        amb_present = (amb_required != 4)
        ambulance_penalty = 0

        if amb_present and action != amb_required:
            ambulance_penalty = self.ambulance_penalty_value

        # ── 8. ambulance persistence — update AFTER compliance check ───
        if self.state_space[9] == 4:
            self.state_space[9] = self.ambulance_phase_generator()
        else:
            if action == self.state_space[9]:
                self.state_space[9] = 4   # ambulance served — clear it

        # ── 9. reward ──────────────────────────────────────────────────
        delta_Q = int(sum(self.state_space[0:8] - prev_queues))

        reward = -(
            delta_Q
            + self.lambda_1 * ambulance_penalty
            + self.lambda_2 * starvation_penalty
        )

        # ── 10. termination ────────────────────────────────────────────
        truncated = self.current_step >= self.max_step
        terminated = False

        info = {
            "ambulance_present":  amb_present,
            "ambulance_required": int(amb_required),
            "ambulance_penalty":  ambulance_penalty,
            "starvation_penalty": starvation_penalty,
            "starvation_counts":  self.phase_starvation_counter.copy(),
            "delta_Q":            delta_Q,
            "queues":             self.state_space[0:8].copy(),
            "action":             action,
            "phase":              self.action_to_phase[action],
        }

        return self.state_space.copy(), reward, terminated, truncated, info