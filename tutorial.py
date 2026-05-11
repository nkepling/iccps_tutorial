# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "gymnasium==1.2.3",
#     "marimo>=0.23.5",
#     "matplotlib==3.10.9",
#     "ns-gym==1.0.12",
#     "numba>=0.60.0",
#     "numpy==2.4.4",
#     "stable-baselines3>=2.3.0",
# ]
# ///


import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"# **CPS-IoT Week 2026 Tutorial — Non-Stationary Decision Making with NS-Gym**"),
            mo.hstack(
                [
                    mo.vstack(
                        [mo.md(r"$$\mathcal{MDP}_1$$"),
                         mo.image("assets/cartpole_stationary.gif", width=260)],
                        align="center",
                    ),
                    mo.md(r"$$\theta_1 \to \theta_2$$"),
                    mo.vstack(
                        [mo.md(r"$$\mathcal{MDP}_2$$"),
                         mo.image("assets/cartpole_nonstationary.gif", width=260)],
                        align="center",
                    ),
                ],
                justify="center",
                align="center",
                gap=2,
            ),
        ],
        gap=2,
    )
    return


@app.cell
def _():
    def punch(text, size="1.4em", color="#1a1a1a", weight=900):
        """Heavier-than-bold span for big-screen visibility."""
        return (
            f'<span style="font-weight: {weight}; font-size: {size}; '
            f'color: {color};">{text}</span>'
        )

    def headline(big, small, size="1.6em", color="#1a1a1a"):
        """Big punchline + smaller body line beneath it."""
        return (
            f'{punch(big, size=size, color=color)}'
            f'<div style="margin-top: 0.4em; opacity: 0.85;">{small}</div>'
        )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goals for this session

    1. **Get hands-on experience** modeling toy non-stationary MDPs with
       NS-Gym and **create your own custom environment.**
    2. **Visually explore** under what conditions "stale" decision-making
       policies break in your environment.
    3. **Evaluate policy performance** using predefined and your own
       custom metrics.
    4. **Evaluate NS-Gym implementations** of decision-making policies
       on your own environment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Agenda  ·  11:00 – 12:30

    | Time | Section |
    |---|---|
    | 11:00 – 11:15 | A quick recap and tutorial material installation |
    | 11:15 – 11:30 | Designing custom toy NS-MDPs |
    | 11:30 – 11:45 | When Fixed Policies Break |
    | 11:45 – 12:00 | Evaluating Algorithms on NS-MDPs |
    | 12:00 – 12:20 | Experimenting with Adaptive Decision Making |
    | 12:20 – 12:30 | Capstone + Question & Discussion |

    > **Color cue.** Purple-bordered cells are lecture / explanation;
    > grey-bordered cells are interactive activities — sliders, code
    > editors, "your turn" prompts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">NS-Gym at a Glance</div></div>

    ### A framework for modeling non-stationary Markov Decision Processes and the key problem types a decision-making entity may encounter:

    - **Design** non-stationary environments in a fully configurable way across a suite of Gymnasium environments.
    - **Emulate** the key problem types for decision making under non-stationarity.
    - **Test and evaluate** decision-making algorithms using included metrics, NS-Gym baseline implementations, or your own RL agents.

    ### The collage below shows the breadth of supported envs — toy text grids, classic control, and MuJoCo continuous control all live behind the same wrapper API.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def _tile(path, label):
        return mo.vstack(
            [mo.image(f"assets/{path}", width=180),
             mo.md(f"<div style='text-align:center; font-size:0.85em; opacity:0.7;'>{label}</div>")],
            align="center",
        )

    mo.vstack([
        mo.md("<div style='text-align:center; font-size:0.9em; opacity:0.6; letter-spacing:0.15em; font-weight:600;'>SUPPORTED ENVIRONMENTS &mdash; TOY TEXT &middot; CLASSIC CONTROL &middot; MUJOCO</div>"),
        mo.hstack([
            _tile("frozen_lake_stationary.gif", "FrozenLake"),
            _tile("cartpole_stationary.gif", "CartPole"),
            _tile("mountain_car.gif", "MountainCar"),
            _tile("pendulum.gif", "Pendulum"),
        ], justify="center", gap=1),
        mo.hstack([
            _tile("ant_stationary.gif", "Ant"),
            _tile("half_cheetah.gif", "HalfCheetah"),
            _tile("hopper.gif", "Hopper"),
            _tile("humanoid.gif", "Humanoid"),
        ], justify="center", gap=1),
        mo.hstack([
            _tile("inverted_pendulum.gif", "InvertedPendulum"),
            _tile("inverted_double_pendulum.gif", "Inv. Double Pend."),
            _tile("reacher.gif", "Reacher"),
            _tile("swimmer.gif", "Swimmer"),
        ], justify="center", gap=1),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Installing the tutorial materials</div></div>
        """),
        mo.image("assets/tutorial_github.png", width=250),
        mo.md(
            "<div style='text-align:center; font-size:0.85em; opacity:0.6;'>"
            "Repo: <a href='https://github.com/nkepling/iccps_tutorial'>"
            "github.com/nkepling/iccps_tutorial</a>."
            "</div>"
        ),
        mo.md(r"""
        ### Three steps. Sandboxed via [uv](https://docs.astral.sh/uv/) — nothing is installed globally.

        ### **1. Install `uv`**

        ```bash
        # macOS / Linux
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Windows (PowerShell)
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

        ### After installing, restart your shell so `uv` is on your `PATH`. Other install methods: <https://docs.astral.sh/uv/getting-started/installation/>.

        ### **2. Clone this repo**

        ```bash
        git clone https://github.com/nkepling/iccps_tutorial.git
        cd iccps_tutorial
        ```

        ### **3. Run the notebook**

        ```bash
        uvx marimo run --sandbox tutorial.py
        ```

        ### `uv` reads the pinned dependency list at the top of `tutorial.py` (PEP 723 inline script metadata), builds an isolated venv, and starts marimo. **First run takes ~30–60 s**; subsequent runs use the cache and start instantly.

        > **Want to edit cells too?** Use `uvx marimo edit --sandbox tutorial.py` instead of `marimo run`.

        ### **Just want NS-Gym in your own project?**

        ```bash
        pip install ns-gym                                       # latest release
        pip install git+https://github.com/scope-lab-vu/ns_gym   # nightly
        ```

        ### Project home: <https://nsgym.io/>  ·  Source: <https://github.com/scope-lab-vu/ns_gym>
        """),
    ], gap=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <div style="
        margin: 5em 0 1.5em 0;
        padding: 2.5em 2em;
        border-top: 8px solid #5a3a8a;
        border-bottom: 8px solid #5a3a8a;
        background: linear-gradient(180deg, #f3eef7, #faf7fd);
    ">
      <div style="font-size: 0.9em; opacity: 0.55;
                  letter-spacing: 0.2em; font-weight: 600;
                  color: #5a3a8a;">
        MODULE 0 &middot; 11:00 · 11:15 &middot; LECTURE
      </div>
      <div style="font-size: 2.7em; font-weight: 800;
                  margin-top: 0.25em; line-height: 1.05;">
        A Quick Recap and Intro to NS-Gym
      </div>
      <div style="font-size: 1.05em; opacity: 0.72;
                  margin-top: 0.7em; max-width: 56em;">
        Quickly review the key ingredients for modeling non-stationary
        sequential decision-making problems, then get the tutorial
        materials installed on your laptop.
      </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Four key questions for decision making in NS-MDPs</div></div>

    ## Every modeling choice in NS-Gym maps onto one of these four questions. Keep them in mind as we go through the rest of the tutorial:

    ## 1. **What** changes?  *(which environmental parameter drifts —slip probability, gravity, masspole, …)*
    ## 2. **How** does it change?  *(the update function — sigmoid, random walk, step, …)*
    ## 3. Does the agent **detect** the change?  *(is there a notificationon every step that the dynamics moved?)*
    ## 4. Does the agent **know** the change?  *(does the notification carry the new parameter, or just a "something moved" flag?)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">The Standard MDP</div></div>

    ### A Markov Decision Process is the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

    - **States** ($\mathcal{S}$) — the agent's "position" in the world.
    -  **Actions** ($\mathcal{A}$) — the choices the agent has at each state. Not all actions are available in all states.
    - **Transitions** $P(s' \mid s, a, \theta)$ — probability over thenext state given the current $(s, a)$ and a parameter $\theta \in \Theta$.
    ###- **Rewards** $R(s, a, s')$ — the instantaneous reward the agent receives when it transitions from $s$ to $s'$ via $a$.

    ### The standard assumption is that **$\theta$ is fixed**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">The non-stationary MDP</div></div>

    ### Following Campo et al. (1991), NS-Gym **disentangles** the Markovian process from the *parameter* uncertainty:

    - A **base MDP** describes the underlying stochastic control problem with parameter $\theta$ held fixed.
    - A **semi-Markov process** on top of $\theta$ describes how the parameter evolves: $\theta_t \xrightarrow{t_s \sim S} \theta_{t+1}$, with $S$ the *scheduler* deciding **when** $\theta$ updates and an *update function* deciding **how**.

    ### This split is what makes NS-Gym composable: any base Gymnasium env pairs with any (scheduler, update-function) recipe.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.hstack([
        mo.vstack([
            mo.image("assets/nsmdp.png", width=620),
            mo.md(
                "<div style='text-align:center; font-size:0.85em; opacity:0.65; max-width:620px;'>"
                "<b>Figure.</b> The agent–environment loop in an NS-MDP. "
                "The base MDP captures endogenous (Markovian) uncertainty; "
                "the parameter process "
                "<i>θ<sub>t</sub> → θ<sub>t+1</sub></i> on top "
                "captures the exogenous, semi-Markov drift of the "
                "transition dynamics."
                "</div>"
            ),
        ], align="center"),
    ], justify="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">What does an "agent" need to operate in non-stationary environments?</div></div>

    ## What constitutes an agent in the literature has been in flux, so in NS-Gym we try to keep the abstraction as generall as possible. We argue that in a real decision making entity may or may not consist of the following elements:

    ### 1. **Decision-making algorithm** — produces an action given the current observation and (a model of) the dynamics.
    ### 2. **Runtime monitor** — answers *"has a parameter changed?" either by listening to the wrapper's notification or by detecting drift on its own.
    ### 3. **Model updater** — answers *"what is the updated parameter?" and refreshes the planning model the decision-maker queries.

    ### NS-Gym can emulate both the runtime and model updater components with its "notifications" mechanism.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.hstack([
        mo.vstack([
            mo.image("assets/env_agent_interaction.png", width=720),
            mo.md(
                "<div style='text-align:center; font-size:0.85em; opacity:0.65; max-width:720px;'>"
                "<b>Figure.</b> Decision-making infrastructure for an "
                "agent operating in non-stationary environments — "
                "the <i>decision-making algorithm</i>, a "
                "<i>runtime monitor</i> (\"has a parameter changed?\"), "
                "and a <i>model updater</i> (\"what is the new "
                "parameter?\"). NS-Gym emulates the runtime monitor "
                "and model updater so you can focus on the "
                "decision-making module."
                "</div>"
            ),
        ], align="center"),
    ], justify="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Three problem types that fall out of those four questions</div></div>

    ### Combining the *what / how / detect / know* questions with the *frequency* of changes carves the literature into a small set of canonical problem types — every NS-MDP paper sits in this 2-D box:

    ### - **Knows-something-changed.** Agent is told $\theta_1$ no longer applies but doesn't see $\theta_2$ — runtime-monitor only.
    ### - **Knows-the-new-model.** Agent is told $\theta_2$, but retraining a deep policy mid-episode is too expensive.
    ### - **Silent drift.** Agent isn't told anything; must detect drift on its own.

    ### Orthogonal axis — *frequency* of change:

    ### 1. **Single shock per episode** — $\theta_1 \to \theta_2$ once.
    ### 2. **Continuous drift within an episode** $\theta_1 \to \theta_2 \to \dots$ many times.
    ### 3. **Between-episode shifts** — every reset gives a new $\theta_0$.

    ### NS-Gym's three notification levels (`none`, `change_notification`, `delta_change_notification`) map directly onto the first axis; the scheduler picks the second.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #5a3a8a; background: #f3eef7;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">A quick intro to the standard Gymnasium API and workflow</div></div>

    ## "Gymnasium" is on open source toolkit for developing decision making algorithms for Markov Decision Procsses

    ## In Gymnasium, an environment represents a Markov Decision Process (MDP). It provides a framework where an agent interacts with an environment by taking actions, observing states, and receiving rewards. Here’s a breakdown of key components:

    ### **Environment** Object: This represents the MDP. It includes a set of states, a set of possible actions, and defines the rules for state transitions and rewards based on the agent's actions.

    ### **Observation**: Represents the information the agent receives about the current state. It may not always provide complete information about the true state, depending on the environment's design.

    ### **Info**: This is an optional dictionary that provides extra diagnostic information helpful for debugging or understanding the environment's behavior. It's not used directly for learning but can provide insights.


    ## Basic Gymasium Ineraction Loop

    ```python
    import random
    import gymnasium as gym

    env = gym.make("FrozenLake-v1") # Make your Environment

    observation, info = env.reset() # Reset it to its initial state

    RandomPolicy = lambda  obs : random.randint(4) # Define your policy

    done = False
    truncated = False

    total_reward = 0

    while not (done or truncated):
        #Given observations decide on an action and "step the environment"
        action = RandomPolicy(obs)

        # Step through your environemt. Receive updated state, and reward signal.
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward

    print(f"TOTAL REWARD: {total_reward}")

    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">How Does NS-Gym Augment Gymnasium? </div></div>

    ## NS-Gym provides a Set of Gymanisium Envionment wrappers that:


    ### 1) Defines a set of observable environment parameters, $\theta$ that in a completely configurable manner be altered to induced non-stationarity

    ### 2) Controls the nature of agent-environment interaction in a non-starionary desicion emulating the runtime and model updater components -- therefore focusing development of the core decision making algorithms
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">The observation NS-Gym hands you</div></div>

    ### Every NS-Gym wrapped env returns a **dictionary** observation — the standard Gymnasium observation lives under `state`, and the wrapper appends three side-channels:

    ```python
    obs = {
        "state":         array([-0.030,  0.197,  0.027, -0.321], dtype=float32),
        "env_change":    {"masscart": 1, "masspole": 0},     # bit per param
        "delta_change":  {"masscart": 0.112, "masspole": 0.0},  # magnitude
        "relative_time": 1,
    }
    ```

    | Field | Always present? | Notification level needed |
    |---|---|---|
    | `state` | ✓ | n/a |
    | `env_change` | only if `change_notification=True` | bit-level |
    | `delta_change` | only if `delta_change_notification=True` | magnitude-level |
    | `relative_time` | ✓ | n/a |

    ### *Pattern.* `env_change[p] == 1` is the "did $\theta$ move?" signal a runtime monitor would otherwise have to compute itself. `delta_change[p]` is the Wasserstein-style magnitude — the model updater's "by how much?" answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Notification levels in action — an execution trace</div></div>

    ### To make the three notification levels concrete, here's a 10-step rollout where the scheduler bumps $\theta$ every two steps. The underlying MDP marches through $\mathcal{M}_0, \mathcal{M}_1, \dots$ regardless of what the agent sees:

    | step | $t_1$ | $t_2$ | $t_3$ | $t_4$ | $t_5$ | $t_6$ | $t_7$ | $t_8$ | $t_9$ | $t_{10}$ |
    |---|---|---|---|---|---|---|---|---|---|---|
    | **MDP** | $\mathcal{M}_0$ | $\mathcal{M}_0$ | $\mathcal{M}_1$ | $\mathcal{M}_1$ | $\mathcal{M}_2$ | $\mathcal{M}_2$ | $\mathcal{M}_3$ | $\mathcal{M}_3$ | $\mathcal{M}_4$ | $\mathcal{M}_4$ |
    | $\theta$ | $\theta_0$ | $\theta_0$ | $\theta_1$ | $\theta_1$ | $\theta_2$ | $\theta_2$ | $\theta_3$ | $\theta_3$ | $\theta_4$ | $\theta_4$ |

    ### Look at the $t_6 \to t_7$ boundary. What does `env.get_planning_env()` give you?

    | `change_notification` | `delta_change_notification` | Agent sees | `get_planning_env()` returns |
    |---|---|---|---|
    | ✗ | ✗ | nothing — only $\mathcal{M}_0$ | snapshot of $\mathcal{M}_0$ |
    | ✓ | ✗ | "something moved" but not $\theta_3$ | still $\mathcal{M}_0$ (last *known* $\theta$) |
    | ✓ | ✓ | exact $\theta_3$ | snapshot of $\mathcal{M}_3$ |

    ### *This is the actual contract you're planning against.* MCTS plans on whatever `get_planning_env()` hands it; if you hide $\theta_3$ it'll plan on $\mathcal{M}_0$ and lose. Same algorithm, different notification level → different behavior.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import gymnasium as gym
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import ns_gym.base as nsg_base
    import ns_gym.envs as _ns_envs
    from ns_gym.envs.Bridge import Bridge as _BridgeCls

    # ns_gym registers `ns_gym/Bridge-v0` with entry point
    # `ns_gym.envs:Bridge`, but `ns_gym/envs/__init__.py` is empty.
    # Importing the submodule above shadows the slot with the *module*
    # object, so gymnasium's `getattr(mod, "Bridge")` returns the module
    # and bombs with "module object is not callable". Force the slot to
    # hold the class instead.
    _ns_envs.Bridge = _BridgeCls

    from ns_gym.schedulers import (
        BurstScheduler,
        ContinuousScheduler,
        CustomScheduler,
        DecayingProbabilityScheduler,
        DiscreteScheduler,
        MemorylessScheduler,
        PeriodicScheduler,
        RandomScheduler,
        WindowScheduler,
    )
    from ns_gym.update_functions import (
        BoundedRandomWalk,
        BudgetBoundedIncrement,
        DeterministicTrend,
        DistributionCyclicUpdate,
        DistributionDecrementUpdate,
        DistributionIncrementUpdate,
        DistributionLinearInterpolation,
        DistributionNoUpdate,
        DistributionStepWiseUpdate,
        ExponentialDecay,
        GeometricProgression,
        IncrementUpdate,
        LCBoundedDistrubutionUpdate,
        OrnsteinUhlenbeck,
        OscillatingUpdate,
        PolynomialTrend,
        RandomCategorical,
        RandomWalk,
        RandomWalkWithDrift,
        RandomWalkWithDriftAndTrend,
        SigmoidTransition,
        TargetReversion,
        UniformDrift,
    )
    from ns_gym.utils import type_mismatch_checker
    from ns_gym.wrappers import (
        NSBridgeWrapper,
        NSClassicControlWrapper,
        NSFrozenLakeWrapper,
    )

    RESULTS_DIR = Path("results")

    def save_sweep(filename, payload):
        """Persist sweep results as JSON for offline post-processing."""
        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = RESULTS_DIR / filename
        serializable = dict(payload)
        if "returns_by_k" in serializable:
            serializable["returns_by_k"] = {
                f"{float(k):.4f}": v
                for k, v in serializable["returns_by_k"].items()
            }
        if "returns_by_k_adaptive" in serializable:
            serializable["returns_by_k_adaptive"] = {
                f"{float(k):.4f}": v
                for k, v in serializable["returns_by_k_adaptive"].items()
            }
        out_path.write_text(json.dumps(serializable, indent=2))
        return str(out_path)

    return (
        BoundedRandomWalk,
        BudgetBoundedIncrement,
        BurstScheduler,
        ContinuousScheduler,
        CustomScheduler,
        DecayingProbabilityScheduler,
        DeterministicTrend,
        DiscreteScheduler,
        DistributionCyclicUpdate,
        DistributionDecrementUpdate,
        DistributionIncrementUpdate,
        DistributionLinearInterpolation,
        DistributionNoUpdate,
        DistributionStepWiseUpdate,
        ExponentialDecay,
        GeometricProgression,
        IncrementUpdate,
        LCBoundedDistrubutionUpdate,
        MemorylessScheduler,
        NSBridgeWrapper,
        NSClassicControlWrapper,
        NSFrozenLakeWrapper,
        OrnsteinUhlenbeck,
        OscillatingUpdate,
        PeriodicScheduler,
        PolynomialTrend,
        RandomCategorical,
        RandomScheduler,
        RandomWalk,
        RandomWalkWithDrift,
        RandomWalkWithDriftAndTrend,
        SigmoidTransition,
        TargetReversion,
        UniformDrift,
        WindowScheduler,
        gym,
        mo,
        np,
        nsg_base,
        plt,
        save_sweep,
        type_mismatch_checker,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <div style="
        margin: 5em 0 1.5em 0;
        padding: 2.5em 2em;
        border-top: 8px solid #1a1a1a;
        border-bottom: 8px solid #1a1a1a;
        background: linear-gradient(180deg, #f5f5f5, #fafafa);
    ">
      <div style="font-size: 0.9em; opacity: 0.55;
                  letter-spacing: 0.2em; font-weight: 600;
                  color: #1a1a1a;">
        MODULE 1 &middot; 11:15 · 11:30
      </div>
      <div style="font-size: 2.7em; font-weight: 800;
                  margin-top: 0.25em; line-height: 1.05;">
        Custom NS-MDPs & When Fixed Policies Fail
      </div>
      <div style="font-size: 1.05em; opacity: 0.72;
                  margin-top: 0.7em; max-width: 56em;">
        Anatomy of an NS-MDP, the gallery of schedulers and drift shapes, and how to combine them into your own non-stationary FrozenLake.
      </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #5a3a8a; background: #f3eef7;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Anatomy of an NS-MDP</div></div>

    ## Every NS-MDP in `ns-gym` is built from four pieces:

    ### 1. A **base Gymnasium environment** — the stationary MDP.
    ### 2. A **scheduler** — *when* does the world change? (returns a bool per step)
    ### 3. An **update function** — *how* does it change? (mutates a parameter)
    ### 4. A **wrapper** — glues them together and exposes change notifications to the agent.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #5a3a8a; background: #f3eef7;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Tunable parameters per environment</div></div>

    ### Every wrapper exposes a fixed set of "hidden θ" knobs you can hook update functions onto. Pick the parameter name from the table below and use it as the key in the `{param: update_fn}` map.

    | Environment | Tunable parameters |
    | --- | --- |
    | Acrobot | `dt`, `LINK_LENGTH_1`, `LINK_LENGTH_2`, `LINK_MASS_1`, `LINK_MASS_2`, `LINK_COM_POS_1`, `LINK_COM_POS_2`, `LINK_MOI` |
    | CartPole | `gravity`, `masscart`, `masspole`, `force_mag`, `tau`, `length` |
    | MountainCar | `force`, `gravity` |
    | Pendulum | `dt`, `g`, `l`, `m` |
    | FrozenLake / CliffWalking | `P` (categorical distribution per state-action) |
    | Bridge | `P`, `left_side_prob`, `right_side_prob` |

    ### Wrappers map onto environment families: `NSClassicControlWrapper`, `NSFrozenLakeWrapper`, `NSCliffWalkingWrapper`, `NSBridgeWrapper`, `MujocoWrapper`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">A 10-line NS-Gym taste-test</div></div>

    ### Straight from the [NS-Gym quickstart](https://nsgym.io/quickstart_guide.html) — a CartPole whose `masspole` drifts continuously and whose `gravity` random-walks every 3 steps:

    ### The basic workflow of NS-Gym easy augments existing Gymnasium environments with non-starionary behavior.

    ### In code the steps are as follows:

    ### **Import Necesarry Modules**

    ```python
    import gymnasium as gym
    from ns_gym.wrappers import NSClassicControlWrapper
    from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
    from ns_gym.update_functions import IncrementUpdate, RandomWalk
    ```

    ### **1)** Make a Gymnasium environment

    ```python
    env = gym.make("CartPole-v1")
    ```

    ### **2)** Decide **what** about then environment changes

    ```python
    param_1 = "masspole" #mass of the pole
    param_2 = "gravity" #gravity of the environment
    ```
    ### **3)** Decide **when** parameters change

    ```python
    scheduler_1 = ContinuousScheduler() #every time step
    scheduler_2 = PeriodicScheduler(period=3) #every third time step
    ```

    ### **4)** Decide **how** parameters change

    ```python
    update_function_1 = IncrementUpdate(sheduler_1, k=0.1) # At scheduler_1 timesteps increment param_1 value by 0.1
    update_function_2 = RandomWalk(scheduler_2) # At scheduler_2 timesteps increment param_2 value by a random walk
    ```
    ### **5)** Map parameter to their change rules

    ```python
    tunable_params = {
        param_1: update_function_1,
        param_2:  update_function_2,
    }
    ```
    ### **6)** Define *how** the agent interacts with the environment**

    ```python
    ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=True)
    obs, info = ns_env.reset()
    ```

    ### Module 1 unpacks every one of these moving pieces — schedulers, update functions, wrappers, notifications — and turns them into sliders so you can build your own.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #5a3a8a; background: #f3eef7;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Stochastic Grid World Environments</div></div>

    ### While NS-Gym can wrap several more complex environments, for this tutorial we will focus on two variations of discrete state and discrete action space stochastic gridworld environments — the **FrozenLake** and the **Bridge** environment.

    1. **Interpretability**: Small state and action spaces make the behavior easier to analyze.
    2. **Solvability**: We can exactly and tractably solve for optimal policies in both stationary and non-stationary cases.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.hstack(
        [
            mo.md(r"""
            **Environment Formalization**

            **States:** Agent $(x, y)$ position.

            **Actions:** Cardinal movement (N, S, E, W).

            **Transitions:** Probabilistic movement based on $P$ chance
            to move as intended; $1 - P$ chance to move perpendicularly.

            **Rewards:** $+1$ at goal state; $0$ otherwise.
            """),
            mo.image("assets/frozenlake_nonstationary.gif", width=320),
        ],
        justify="center",
        align="center",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <div style="
        margin: 5em 0 1.5em 0;
        padding: 2em 2em;
        border-top: 6px solid #5a3a8a;
        border-bottom: 6px solid #5a3a8a;
        background: linear-gradient(180deg, #f3eef7, #faf7fd);
    ">
      <div style="font-size: 0.9em; opacity: 0.55;
                  letter-spacing: 0.2em; font-weight: 600;
                  color: #5a3a8a;">
        INTERLUDE &middot; Mathematical preliminaries
      </div>
      <div style="font-size: 3em; font-weight: 800;
                  margin-top: 0.25em; line-height: 1.05;">
        Value iteration, oracle VI, and what we'll measure
      </div>
      <div style="font-size: 1em; opacity: 0.72;
                  margin-top: 0.7em; max-width: 56em;">
        Three planners whose behavior we will compare for the rest of
        the tutorial. Defining them once now so the demos and metrics
        in the next three modules don't need to re-explain the math.
      </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">1. Stationary value iteration</div></div>

    ### **Value of a state** is the long-run expected discounted reward, starting from $s$ and following some policy.

    ### For a stationary MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$, the value function is computed via the Bellman equations:

    $$
    V^*(s) \;=\; \max_a \sum_{s'} P(s' \!\mid\! s, a)\,\bigl[R + \gamma\, V^*(s')\bigr]
    $$

    $$
    V^\pi(s) \;=\; \sum_{s'} P\bigl(s' \!\mid\! s,\, \pi(s)\bigr)\,\bigl[R + \gamma\, V^\pi(s')\bigr]
    $$

    ### **$V^*$ vs $V^\pi$.**  $V^*$ is the *ceiling* — what the best policy gets. $V^\pi$ is what you actually get running policy $\pi$. The only difference is the $\max_a$ in the top backup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">1b. Action-value (Q) function</div></div>

    ### Similarly, the **action-value function** $Q^*(s, a)$ is the expected long-run discounted reward from taking action $a$ at state $s$ and then acting optimally.

    $$
    Q^*(s, a) = \sum_{s'} P(s' \mid s, a)\,\bigl[R(s, a, s') + \gamma\, V^*(s')\bigr]
    $$

    ### Connection: $\;V^*(s) = \max_a Q^*(s, a)\;$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">2. Three planners we will compare</div></div>

    | Planner | Plans against | Re-solves? | Knows future? |
    |---|---|---|---|
    | **Stale VI** | $P_0$ (initial transitions) | never | no |
    | **Myopic replan VI** | current $P_t$ at each step | every step (or on notification) | no |
    | **Oracle VI** | full $\{P_0, P_1, \dots, P_{T-1}\}$ | once, offline | yes (assumes the schedule is known) |

    ### All three solve the same Bellman fixed point — they differ only in the model they're handed. Stale never updates. Myopic re-solves on the fly using the latest snapshot (`env.get_planning_env()`). Oracle sees the entire $P_t$ trajectory in advance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">3. Oracle / time-augmented VI</div></div>

    ### Given a finite horizon $T$ and the full transition schedule $\{P_t\}_{t=0}^{T-1}$, the **oracle** solves the time-augmented Bellman equation by **backward induction**:

    $$
    V^*(s, T) = 0
    $$

    $$
    V^*(s, t) = \max_a \sum_{s'} P_t(s' \mid s, a)\,\bigl[R_t(s, a, s') + \gamma\, V^*(s', t{+}1)\bigr]
    $$

    ### **The ceiling.** Cheats by reading the future — no agent can beat it in expectation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">4. The metrics we will track</div></div>

    ### **Cumulative reward** — the realized return of an episode.

    $$
    G \;=\; \sum_{t=0}^{T-1} \gamma^t R_t
    $$

    ### *Tells us:* raw performance under non-sationarity

    ---

    ### **Q-gap** *(a.k.a. dynamic regret — Cheung, Simchi-Levi & Zhu, ICML 2020)* — per-step value loss against the time-indexed oracle.

    $$
    \Delta_Q(s_t, t) \;=\; Q^*_t(s_t, a^*_t) \;-\; Q^*_t(s_t, a_{\text{taken},t})
    $$

    ### *Tells us:* how much each step costs you in expected return — cumulative $\Delta_Q$ is regret against the oracle ceiling.

    ---

    ### **Action gap** — decisiveness of the optimal action: best $Q^*$ minus second-best, at $(s, t)$.

    $$
    \text{gap}(s, t) \;=\; \max_a Q^*_t(s, a) \;-\; \underset{a}{\text{2nd-max}}\, Q^*_t(s, a)
    $$

    ### *Tells us:* How important is ti to

    > **Reference.** Cheung, Simchi-Levi & Zhu, *Reinforcement Learning for Non-Stationary Markov Decision Processes: The Blessing of (More) Optimism*, ICML 2020, [PMLR 119:1843–1854](https://proceedings.mlr.press/v119/cheung20a.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Non-Stationary FrozenLake</div></div>

    ### The remainder of the session will use **FrozenLake** as our running example — both for designing a non-stationary version of the environment and for examining how a policy can break under those conditions.
    """)
    return


@app.function(hide_code=True)
def init_dist(p_intended):
    """Categorical [P_intended, (1-P)/2, (1-P)/2] from a scalar p ∈ [0, 1]."""
    p = float(p_intended)
    rest = (1.0 - p) / 2.0
    return [p, rest, rest]


@app.cell(hide_code=True)
def _(mo):
    fl_init_intended = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=1.0,
        label="FrozenLake — initial P[intended]",
    )
    mo.vstack([
        mo.md(r"""
        <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #555; background: #f6f6f6;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">FrozenLake — initial transition probability</div></div>

        ### Set the **initial** `P[intended]`. Remaining mass is split evenly across the two perpendicular slip directions: `[p, (1-p)/2, (1-p)/2]`. The scheduler + update function below modify this distribution over time as the rollout progresses.

        - **1.0** → deterministic at $t=0$ (no slip yet)
        - **0.5** → moderate slip from the start
        - **0.0** → no progress in the intended direction; perpendicular slip only
        """),
        fl_init_intended,
    ])
    return (fl_init_intended,)


@app.cell
def _(
    ContinuousScheduler,
    DistributionDecrementUpdate,
    NSFrozenLakeWrapper,
    fl_init_intended,
    gym,
):
    def fl_make_env(k):
        env = gym.make("FrozenLake-v1", is_slippery=False, max_episode_steps=50)
        scheduler = ContinuousScheduler()
        update_fn = DistributionDecrementUpdate(scheduler=scheduler, k=float(k))
        return NSFrozenLakeWrapper(
            env,
            {"P": update_fn},
            change_notification=True,
            delta_change_notification=True,
            initial_prob_dist=init_dist(fl_init_intended.value),
        )

    return (fl_make_env,)


@app.cell
def _(np, type_mismatch_checker):
    def fl_rollout(env, policy_fn, seed=0):
        obs, _ = env.reset(seed=seed)
        state, _ = type_mismatch_checker(obs)
        done = trunc = False
        intended_prob = []
        deltas = []
        ep_return = 0.0
        while not (done or trunc):
            action = policy_fn(state)
            obs, reward, done, trunc, _ = env.step(action)
            state, reward = type_mismatch_checker(obs, reward)
            intended_prob.append(float(env.transition_prob[0]))
            deltas.append(float(obs.get("delta_change", {}).get("P", 0.0)))
            ep_return += reward
        return ep_return, intended_prob, deltas

    fl_rng = np.random.default_rng(0)

    def fl_random_policy(_state):
        return int(fl_rng.integers(0, 4))

    return fl_random_policy, fl_rollout


@app.cell
def _():
    """FrozenLake-v1 4×4 layout + action arrows.

    Shared by every FL / Bridge heatmap renderer. Lives in Module 1
    since both gridworlds use the same action-arrow glyphs.
    """
    FL_MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]
    FL_ACTION_ARROWS = ["←", "↓", "→", "↑"]
    return FL_ACTION_ARROWS, FL_MAP


@app.cell
def _(np):
    """Plain-Python value iteration for any tabular Gymnasium env.

    Returns ``(policy, V)``. We solve once on the *stationary*
    FrozenLake (no slip, gamma=0.95) to produce the baseline
    `fl_stationary_V` / `fl_stationary_policy` arrays consumed
    downstream as the "stale" planner and the t=0 anchor for the
    myopic-replan agent.
    """
    def value_iteration(env, gamma=0.95, theta=1e-6, max_iter=2000):
        n_states = env.unwrapped.observation_space.n
        n_actions = env.unwrapped.action_space.n
        V = np.zeros(n_states)
        for _ in range(max_iter):
            delta = 0.0
            for s in range(n_states):
                v_old = V[s]
                V[s] = max(
                    sum(p * (r + gamma * V[s2])
                        for p, s2, r, _ in env.unwrapped.P[s][a])
                    for a in range(n_actions)
                )
                delta = max(delta, abs(v_old - V[s]))
            if delta < theta:
                break
        policy = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            policy[s] = int(np.argmax([
                sum(p * (r + gamma * V[s2])
                    for p, s2, r, _ in env.unwrapped.P[s][a])
                for a in range(n_actions)
            ]))
        return policy, V

    return (value_iteration,)


@app.cell
def _(gym, value_iteration):
    """Stationary baseline: solve FrozenLake exactly once on the
    no-slip MDP. The resulting V / policy are referenced by every
    "stale-VI" panel and used as the t=0 anchor for the myopic-replan
    agent."""
    fl_stationary_env = gym.make(
        "FrozenLake-v1", is_slippery=False, max_episode_steps=50,
    )
    fl_stationary_policy, fl_stationary_V = value_iteration(
        fl_stationary_env,
    )
    return fl_stationary_V, fl_stationary_policy


@app.cell
def _(fl_stationary_policy):
    """Deployment-time policy callable: maps state → action by
    indexing the stationary policy table. Used by every comparison
    cell that calls ``fl_rollout(env, fl_vi_policy, ...)``."""
    def fl_vi_policy(state):
        return int(fl_stationary_policy[int(state)])

    return (fl_vi_policy,)


@app.cell(hide_code=True)
def _(mo):
    fl_sched_pick = mo.ui.dropdown(
        options=[
            "ContinuousScheduler — fires every step",
            "PeriodicScheduler(period=5)",
            "DiscreteScheduler — explicit set {8..14, 30..34}",
            "RandomScheduler(probability=0.25)",
            "MemorylessScheduler(p=0.2) — geometric inter-arrivals",
            "BurstScheduler(on=5, off=10)",
            "DecayingProbabilityScheduler(p₀=0.8, λ=0.1)",
            "WindowScheduler(windows=[(5,10),(20,25),(35,45)])",
            "CustomScheduler(λ t: t² mod 7 == 0)",
            "★ Your custom (FLUserScheduler)",
        ],
        value="ContinuousScheduler — fires every step",
        label="scheduler",
    )
    mo.vstack([
        mo.md(r"""
        <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Schedule gallery — *when* does the world change?</div></div>

        ### Pick a scheduler and watch when it fires over a 50-step window. Same parameter, very different drift trajectories. No code required — just explore.
        """),
        fl_sched_pick,
    ])
    return (fl_sched_pick,)


@app.cell
def _(
    BurstScheduler,
    ContinuousScheduler,
    CustomScheduler,
    DecayingProbabilityScheduler,
    DiscreteScheduler,
    MemorylessScheduler,
    PeriodicScheduler,
    RandomScheduler,
    WindowScheduler,
    fl_sched_pick,
    fl_user_scheduler_cls,
):
    def fl_make_gallery_scheduler():
        """Build a *fresh* scheduler matching the dropdown selection.

        Returns a new instance each call so internal state (RNG, next-event
        time, etc.) is not shared across env instances. If the user picks
        **★ Your custom (FLUserScheduler)** but their class hasn't
        compiled yet, we fall back to ``ContinuousScheduler`` so the
        downstream pipeline still produces a result.
        """
        v = fl_sched_pick.value
        if v.startswith("★ Your custom"):
            if fl_user_scheduler_cls is None:
                # Fallback so the rest of the panels still render.
                return ContinuousScheduler()
            return fl_user_scheduler_cls()
        if v.startswith("ContinuousScheduler"):
            return ContinuousScheduler()
        if v.startswith("PeriodicScheduler"):
            return PeriodicScheduler(period=5)
        if v.startswith("DiscreteScheduler"):
            return DiscreteScheduler(
                event_list=set(range(8, 15)) | set(range(30, 35))
            )
        if v.startswith("RandomScheduler"):
            return RandomScheduler(probability=0.25, seed=0)
        if v.startswith("MemorylessScheduler"):
            return MemorylessScheduler(p=0.2, seed=0)
        if v.startswith("BurstScheduler"):
            return BurstScheduler(on_duration=5, off_duration=10)
        if v.startswith("DecayingProbabilityScheduler"):
            return DecayingProbabilityScheduler(
                initial_probability=0.8, decay_rate=0.1, seed=0,
            )
        if v.startswith("WindowScheduler"):
            return WindowScheduler(windows=[(5, 10), (20, 25), (35, 45)])
        return CustomScheduler(event_function=lambda t: (t * t) % 7 == 0)

    return (fl_make_gallery_scheduler,)


@app.cell
def _(fl_make_gallery_scheduler, fl_sched_pick, np, plt):
    _T = 50
    _sched_for_plot = fl_make_gallery_scheduler()
    _fires = np.array([int(_sched_for_plot(t)) for t in range(_T)])
    _fig, _ax = plt.subplots(figsize=(8, 1.6))
    _ax.bar(range(_T), _fires, color="tab:red")
    _ax.set_yticks([0, 1])
    _ax.set_xlabel("step")
    _ax.set_ylabel("fires?")
    _ax.set_title(
        f"{fl_sched_pick.value}   "
        f"({int(_fires.sum())}/{_T} = {_fires.mean():.0%} fires)"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    DRIFT_OPTIONS = [
        # — designed to expose oracle vs myopic divergence —
        "★ SigmoidEarlyCliff(t0=4, k=1.0) — oracle ≠ myopic",
        "★ DistributionDecrementUpdate(k=0.15) — sharp linear decay",
        "★ Your custom (FLUserUpdate)",
        # — native distribution updates —
        "DistributionNoUpdate — identity",
        "DistributionDecrementUpdate(k=0.05)",
        "DistributionIncrementUpdate(k=0.05)",
        "RandomCategorical — fresh random each fire",
        "UniformDrift(rate=0.05) — toward uniform",
        "TargetReversion(target=[0.5,0.3,0.2], θ=0.1)",
        "DistributionLinearInterpolation([1,0,0]→[1/3,1/3,1/3], T=50)",
        "DistributionCyclicUpdate — cycles 4 dists",
        "DistributionStepWiseUpdate — 3 step values",
        "LCBoundedDistrubutionUpdate(L=0.3)",
        "BudgetBoundedIncrement(k=0.05, B=2.0)",
        # — scalar updates lifted via ScalarToCategorical (Pattern A) —
        "↑ DeterministicTrend(slope=-0.01)",
        "↑ PolynomialTrend(coeffs=[-0.005, -0.0001])",
        "↑ SigmoidTransition(a=1, b=0, k=0.3, t0=25)",
        "↑ OscillatingUpdate(delta=0.05)",
        "↑ ExponentialDecay(decay_rate=0.05)",
        "↑ GeometricProgression(r=0.95)",
        "↑ OrnsteinUhlenbeck(θ=0.1, μ=0.5, σ=0.05)",
        "↑ RandomWalk(μ=0, σ=0.05)",
        "↑ RandomWalkWithDrift(α=-0.01, μ=0, σ=0.03)",
        "↑ RandomWalkWithDriftAndTrend(α=-0.005, μ=0, σ=0.02, slope=0.0005)",
        "↑ BoundedRandomWalk(μ=0, σ=0.08, lo=0, hi=1)",
    ]
    return (DRIFT_OPTIONS,)


@app.cell(hide_code=True)
def _(DRIFT_OPTIONS, mo):
    fl_drift_pick = mo.ui.dropdown(
        options=DRIFT_OPTIONS,
        value="★ SigmoidEarlyCliff(t0=4, k=1.0) — oracle ≠ myopic",
        label="drift shape  (★ = tuned for visible divergence; ↑ = scalar lifted)",
    )
    mo.vstack([
        mo.md(r"""
        <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Drift-shape gallery — how does the parameter change?</div></div>

        ### Pick a categorical-distribution update function and watch the intended-direction probability `P[intended]` evolve over 50 steps of a `ContinuousScheduler`. Same scheduler, very different drift shapes.
        """),
        fl_drift_pick,
    ])
    return (fl_drift_pick,)


@app.cell
def _(nsg_base):
    class ScalarToCategorical(nsg_base.UpdateDistributionFn):
        """Lift any scalar update to a categorical via Pattern A:
        apply scalar update to param[0] (intended-direction prob),
        clip to [0, 1], split (1 - param[0]) uniformly across the rest.

        We bypass the inner update's __call__ and invoke its `_update`
        directly so the inner scheduler doesn't double-fire.
        """

        def __init__(self, scheduler, scalar_update_cls, **kwargs):
            super().__init__(scheduler)
            self._inner = scalar_update_cls(scheduler=scheduler, **kwargs)

        def _update(self, param, t):
            new0 = float(self._inner._update(param[0], t))
            new0 = max(0.0, min(1.0, new0))
            rest = (1.0 - new0) / max(len(param) - 1, 1)
            return [new0] + [rest] * (len(param) - 1)

    return (ScalarToCategorical,)


@app.cell
def _(
    BoundedRandomWalk,
    BudgetBoundedIncrement,
    DeterministicTrend,
    DistributionCyclicUpdate,
    DistributionDecrementUpdate,
    DistributionIncrementUpdate,
    DistributionLinearInterpolation,
    DistributionNoUpdate,
    DistributionStepWiseUpdate,
    ExponentialDecay,
    GeometricProgression,
    LCBoundedDistrubutionUpdate,
    OrnsteinUhlenbeck,
    OscillatingUpdate,
    PolynomialTrend,
    RandomCategorical,
    RandomWalk,
    RandomWalkWithDrift,
    RandomWalkWithDriftAndTrend,
    ScalarToCategorical,
    SigmoidTransition,
    TargetReversion,
    UniformDrift,
    fl_drift_pick,
    fl_user_update_cls,
):
    def fl_make_gallery_update(scheduler, drift_value=None):
        """Build a *fresh* update fn matching the dropdown, bound to `scheduler`.

        If `drift_value` is provided it overrides `fl_drift_pick.value` —
        useful when a different dropdown (e.g. Bridge per-side) should
        drive the choice. If the user picks **★ Your custom
        (FLUserUpdate)** but their class hasn't compiled yet, fall
        back to ``DistributionNoUpdate`` so the rest of the pipeline
        still renders.

        Returns a new instance each call so any internal state (RNG, cycle
        index, total-change accumulator, …) is not shared across env
        instances.
        """
        v = drift_value if drift_value is not None else fl_drift_pick.value

        if v.startswith("★ Your custom"):
            if fl_user_update_cls is None:
                return DistributionNoUpdate(scheduler)
            # User's FLUserUpdate signature is (scheduler, T=20). We
            # don't know if they took the T kwarg or not, so try with
            # T first, fall back to plain (scheduler).
            try:
                return fl_user_update_cls(scheduler, T=20)
            except TypeError:
                return fl_user_update_cls(scheduler)

        # Tuned for visible oracle vs myopic divergence ------------------
        # NB: these checks must come first because they share prefixes
        # with the native entries below.
        if v.startswith("★ SigmoidEarlyCliff"):
            # P[intended] drops sharply from 1→0 around step 4
            # (σ(0)≈0.98, σ(4)=0.5, σ(6)≈0.12, σ(8)≈0.02). Oracle
            # races to the goal in ≤5 steps along a robust path; myopic
            # at t=0 plans deterministic and may pick a path that gets
            # destroyed by the cliff.
            return ScalarToCategorical(
                scheduler, SigmoidTransition,
                a=1.0, b=0.0, k=1.0, t0=4,
            )
        if v.startswith("★ DistributionDecrementUpdate"):
            # k=0.15 hits intended=0 around t=7. Goal is 6 steps away,
            # so oracle has to pick a path that's *both* fast and
            # robust to the rapidly-rising slip.
            return DistributionDecrementUpdate(scheduler=scheduler, k=0.15)

        # Native distribution updates -----------------------------------
        if v.startswith("DistributionNoUpdate"):
            return DistributionNoUpdate(scheduler)
        if v.startswith("DistributionDecrementUpdate"):
            return DistributionDecrementUpdate(scheduler=scheduler, k=0.05)
        if v.startswith("DistributionIncrementUpdate"):
            return DistributionIncrementUpdate(scheduler=scheduler, k=0.05)
        if v.startswith("RandomCategorical"):
            return RandomCategorical(scheduler, seed=0)
        if v.startswith("UniformDrift"):
            return UniformDrift(scheduler, rate=0.05)
        if v.startswith("TargetReversion"):
            return TargetReversion(
                scheduler, target=[0.5, 0.3, 0.2], theta=0.1,
            )
        if v.startswith("DistributionLinearInterpolation"):
            return DistributionLinearInterpolation(
                scheduler,
                start_dist=[1.0, 0.0, 0.0],
                end_dist=[1 / 3, 1 / 3, 1 / 3],
                T=50,
            )
        if v.startswith("DistributionCyclicUpdate"):
            return DistributionCyclicUpdate(
                scheduler,
                dist_list=[
                    [1.0, 0.0, 0.0],
                    [0.5, 0.25, 0.25],
                    [0.2, 0.4, 0.4],
                    [1 / 3, 1 / 3, 1 / 3],
                ],
            )
        if v.startswith("DistributionStepWiseUpdate"):
            return DistributionStepWiseUpdate(
                scheduler,
                update_values=[
                    [0.8, 0.1, 0.1],
                    [0.5, 0.25, 0.25],
                    [0.2, 0.4, 0.4],
                ],
            )
        if v.startswith("LCBoundedDistrubutionUpdate"):
            return LCBoundedDistrubutionUpdate(scheduler, L=0.3)
        if v.startswith("BudgetBoundedIncrement"):
            return BudgetBoundedIncrement(scheduler, k=0.05, B=2.0)

        # Scalar updates lifted via ScalarToCategorical (Pattern A) -----
        if v.startswith("↑ DeterministicTrend"):
            return ScalarToCategorical(scheduler, DeterministicTrend, slope=-0.01)
        if v.startswith("↑ PolynomialTrend"):
            return ScalarToCategorical(
                scheduler, PolynomialTrend, coeffs=[-0.005, -0.0001],
            )
        if v.startswith("↑ SigmoidTransition"):
            return ScalarToCategorical(
                scheduler, SigmoidTransition, a=1.0, b=0.0, k=0.3, t0=25,
            )
        if v.startswith("↑ OscillatingUpdate"):
            return ScalarToCategorical(scheduler, OscillatingUpdate, delta=0.05)
        if v.startswith("↑ ExponentialDecay"):
            return ScalarToCategorical(
                scheduler, ExponentialDecay, decay_rate=0.05,
            )
        if v.startswith("↑ GeometricProgression"):
            return ScalarToCategorical(scheduler, GeometricProgression, r=0.95)
        if v.startswith("↑ OrnsteinUhlenbeck"):
            return ScalarToCategorical(
                scheduler, OrnsteinUhlenbeck,
                theta=0.1, mu=0.5, sigma=0.05, seed=0,
            )
        if v.startswith("↑ RandomWalkWithDriftAndTrend"):
            return ScalarToCategorical(
                scheduler, RandomWalkWithDriftAndTrend,
                alpha=-0.005, mu=0.0, sigma=0.02, slope=0.0005, seed=0,
            )
        if v.startswith("↑ RandomWalkWithDrift"):
            return ScalarToCategorical(
                scheduler, RandomWalkWithDrift,
                alpha=-0.01, mu=0.0, sigma=0.03, seed=0,
            )
        if v.startswith("↑ RandomWalk"):
            return ScalarToCategorical(
                scheduler, RandomWalk, mu=0.0, sigma=0.05, seed=0,
            )
        # ↑ BoundedRandomWalk
        return ScalarToCategorical(
            scheduler, BoundedRandomWalk,
            mu=0.0, sigma=0.08, lo=0.0, hi=1.0, seed=0,
        )

    return (fl_make_gallery_update,)


@app.cell
def _(ContinuousScheduler, fl_drift_pick, fl_make_gallery_update, plt):
    # For the trajectory plot we use ContinuousScheduler so the *shape* of
    # the drift is visible regardless of which scheduler is selected.
    _gallery_update_for_plot = fl_make_gallery_update(ContinuousScheduler())
    _T = 50
    _param = [1.0, 0.0, 0.0]
    _intended = []
    _deltas = []
    for _t in range(_T):
        _param, _fired, _delta = _gallery_update_for_plot(_param, _t)
        _intended.append(float(_param[0]))
        _deltas.append(float(_delta))

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 2.6))
    _axes[0].plot(_intended, marker="o", markersize=3)
    _axes[0].set_xlabel("step")
    _axes[0].set_ylabel("P[intended]")
    _axes[0].set_ylim(-0.05, 1.05)
    _axes[0].set_title(f"{fl_drift_pick.value}")
    _axes[1].bar(range(_T), _deltas)
    _axes[1].set_xlabel("step")
    _axes[1].set_ylabel("Wasserstein delta")
    _axes[1].set_title("per-step delta")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Your turn — custom scheduler</div></div>

    ### Subclass `nsg_base.Scheduler` and implement your own scheduler. Implement `_check(t) -> bool`. Where it returns true at time

    > **Where your code is reused.** Once `FLUserScheduler` compiles
    > (look for the ✓ below), it's available everywhere:
    >
    > 1. Pick **★ Your custom (FLUserScheduler)** in the scheduler
    >    dropdown above to drive the **Your NS-MDP** combiner panel,
    >    heatmaps, and performance bars further down.
    > 2. The **"Demo — full pipeline with your code"** cell directly
    >    below runs all three planners end-to-end against your
    >    `FLUserScheduler` + `FLUserUpdate` combo.
    > 3. The **Capstone editor** has a ` Your saved scheduler/update`
    >    reference tab that loads your editor contents as starter code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _stub = """class FLUserScheduler(nsg_base.Scheduler):
    \"\"\"Custom scheduler — fire with probability 0.25 each step.\"\"\"
    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng(0)

    def _check(self, t):
        # TODO: return True with probability 0.25 at each step
        return False
    """
    _solution = """class FLUserScheduler(nsg_base.Scheduler):
    \"\"\"Reference: Bernoulli(0.25) firing pattern.\"\"\"
    def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(0)

        def _check(self, t):
            return bool(self.rng.random() < 0.25)
    """
    fl_scheduler_editor = mo.ui.code_editor(
        value=_stub, language="python", min_height=10,
    )
    _ref = mo.ui.code_editor(
        value=_solution, language="python", disabled=True, min_height=10,
    )
    mo.ui.tabs({
        "✏️ Your code": fl_scheduler_editor,
        "💡 Reference solution": _ref,
    })
    return (fl_scheduler_editor,)


@app.cell
def _(fl_scheduler_editor, np, nsg_base):
    _ns = {"np": np, "nsg_base": nsg_base}
    try:
        exec(fl_scheduler_editor.value, _ns)
        fl_user_scheduler_cls = _ns.get("FLUserScheduler")
        if fl_user_scheduler_cls is None:
            raise NameError("define a class named FLUserScheduler")
        _msg = "✓ scheduler compiled — now available as `fl_user_scheduler_cls`"
    except Exception as _e:
        fl_user_scheduler_cls = None
        _msg = f"⚠ {type(_e).__name__}: {_e}"
    _msg
    return (fl_user_scheduler_cls,)


@app.cell
def _(fl_user_scheduler_cls):
    if fl_user_scheduler_cls is None:
        _out = "⚠ Fix the class definition above to see results."
    else:
        _sched = fl_user_scheduler_cls()
        fl_user_fire_count = sum(int(_sched(_t)) for _t in range(300))
        _out = f"Custom scheduler fired {fl_user_fire_count}/300 = {fl_user_fire_count / 300:.2%} of the time"
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Your turn — custom update function</div></div>

    ### Schedulers say *when* the world changes. Update functions say *how*. Subclass `nsg_base.UpdateDistributionFn` and write your own. The verification cell below builds a FrozenLake env using your update function, rolls 30 steps under a random policy, and plots `P[intended]` over time. If your ramp is correct you should see a clean diagonal from 1 → 0.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    from textwrap import dedent as _dedent
    _stub = _dedent('''\
        class FLUserUpdate(nsg_base.UpdateDistributionFn):
            """Linear ramp: P[intended] decays 1 → 0 over T steps."""
            def __init__(self, scheduler, T=20):
                super().__init__(scheduler)
                self.T = T

            def _update(self, param, t):
                # param is a 3-list: [P[intended], P[slip-left], P[slip-right]]
                # TODO: return the same shape with
                #         P[intended] = max(0, 1 - t / self.T)
                #         remaining mass split evenly between the two slips.
                return param
        ''')
    _solution = _dedent('''\
        class FLUserUpdate(nsg_base.UpdateDistributionFn):
            """Reference: linear ramp on P[intended]."""
            def __init__(self, scheduler, T=20):
                super().__init__(scheduler)
                self.T = T

            def _update(self, param, t):
                p_int = max(0.0, 1.0 - t / self.T)
                slip = (1.0 - p_int) / 2.0
                return [p_int, slip, slip]
        ''')
    fl_update_editor = mo.ui.code_editor(
        value=_stub, language="python", min_height=12,
    )
    _ref = mo.ui.code_editor(
        value=_solution, language="python", disabled=True, min_height=12,
    )
    mo.ui.tabs({
        "✏️ Your code": fl_update_editor,
        "💡 Reference solution": _ref,
    })
    return (fl_update_editor,)


@app.cell
def _(fl_update_editor, np, nsg_base):
    _ns = {"np": np, "nsg_base": nsg_base}
    try:
        exec(fl_update_editor.value, _ns)
        fl_user_update_cls = _ns.get("FLUserUpdate")
        if fl_user_update_cls is None:
            raise NameError("define a class named FLUserUpdate")
        _msg = "✓ FLUserUpdate compiled — now available as `fl_user_update_cls`"
    except Exception as _e:
        fl_user_update_cls = None
        _msg = f"⚠ {type(_e).__name__}: {_e}"
    _msg
    return (fl_user_update_cls,)


@app.cell
def _(
    ContinuousScheduler,
    NSFrozenLakeWrapper,
    fl_user_update_cls,
    gym,
    mo,
    np,
    plt,
):
    if fl_user_update_cls is None:
        _out = mo.md("⚠ Fix the class definition above to see the ramp plot.")
    else:
        try:
            _base = gym.make(
                "FrozenLake-v1", is_slippery=False, max_episode_steps=50,
            )
            _T = 20
            _upd = fl_user_update_cls(ContinuousScheduler(), T=_T)
            _env = NSFrozenLakeWrapper(
                _base, {"P": _upd},
                change_notification=True,
                delta_change_notification=True,
                initial_prob_dist=[1.0, 0.0, 0.0],
            )
            _env.reset(seed=0)
            _traj = [float(_env.transition_prob[0])]
            np.random.seed(0)
            _done = _trunc = False
            _t = 0
            while not (_done or _trunc) and _t < 30:
                _a = int(_env.action_space.sample())
                _, _, _done, _trunc, _ = _env.step(_a)
                _traj.append(float(_env.transition_prob[0]))
                _t += 1
            _fig, _ax = plt.subplots(figsize=(7, 2.6), layout="constrained")
            _ax.plot(_traj, marker="o", markersize=3, color="tab:blue",
                     linewidth=1.6, label="your update_fn")
            _ax.plot(
                [max(0.0, 1.0 - i / _T) for i in range(len(_traj))],
                color="tab:gray", linewidth=1.0, linestyle="--",
                label="reference ramp (T=20)",
            )
            _ax.set_xlabel("step")
            _ax.set_ylabel("P[intended]")
            _ax.set_title(
                "Custom update fn — does P[intended] follow a clean ramp?",
                fontsize=10,
            )
            _ax.set_ylim(-0.05, 1.05)
            _ax.legend(fontsize=8, loc="upper right")
            _out = _fig
        except Exception as _e:
            _out = mo.md(f"⚠ rollout failed: `{type(_e).__name__}: {_e}`")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Your turn — build & register your own NS-MDP</div></div>

    ### You've written a scheduler and an update function. Now wire them into a **full non-stationary env** and **register it with Gymnasium** so it can be loaded anywhere via `gym.make("YourEnvId-v0")`

    The contract is three things:

    1. **A factory** — a `make_env(**kwargs)` function that builds and returns the wrapped env. The factory must accept `change_notification` / `delta_change_notification` kwargs so callers can control the agent's visibility into drift.
    2. **A wrapper** — `NSFrozenLakeWrapper` here (or `NSClassicControlWrapper` / `NSBridgeWrapper` / `NSMujocoWrapper` for other base envs). Pair each tunable parameter name with its update function in a dict.
    3. **A registration call** — `register(id=..., entry_point=make_env, disable_env_checker=True, order_enforce=False)`. After this, `gym.make("YourEnvId-v0")` works exactly like a built-in env.

    > **Reuse.** The stub starts with **your** `FLUserScheduler` and
    > `FLUserUpdate` from above (available as `fl_user_scheduler_cls` /
    > `fl_user_update_cls` in the exec sandbox). Swap them for any
    > `ns_gym.schedulers` / `ns_gym.update_functions` pair you like.
    >
    > **Registration semantics.** `register()` mutates a global
    > registry — re-running this cell after edits is safe; it
    > overwrites the previous registration (gymnasium emits a
    > `UserWarning` and moves on). The variable `fl_register_env_id`
    > flows downstream so the verifier cell below can call
    > `gym.make(fl_register_env_id)`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    from textwrap import dedent as _dedent
    _stub = _dedent('''\
        # 1) Imports
        import gymnasium as gym
        from gymnasium.envs.registration import register
        from ns_gym.wrappers import NSFrozenLakeWrapper
        from ns_gym.schedulers import ContinuousScheduler
        # from ns_gym.update_functions import ...   # ← uncomment / extend

        # 2) Factory — gymnasium will call this when someone does
        #    gym.make("YourEnvId-v0", **kwargs).
        def make_env(**kwargs):
            change_notification = kwargs.get("change_notification", True)
            delta_change_notification = kwargs.get(
                "delta_change_notification", True,
            )

            #TODO: Make your base environment

            # TODO: build a scheduler.

            # TODO: build an update fn paired to that scheduler.

            #TODO
            return NSFrozenLakeWrapper(
                base, tunable_params,
                change_notification=change_notification,
                delta_change_notification=delta_change_notification,
                initial_prob_dist=[1.0, 0.0, 0.0],
            )

        # 3) Pick an env id and register the factory.
        ENV_ID = "MyNSFrozenLake-v0"
        register(
            id=ENV_ID,
            entry_point=make_env,
            disable_env_checker=True,
            order_enforce=False,
        )
        ''')
    _solution = _dedent('''\
        # Reference — wires your FLUserScheduler + FLUserUpdate from
        # the two exercises above into a registered NS-FrozenLake env.
        import gymnasium as gym
        from gymnasium.envs.registration import register
        from ns_gym.wrappers import NSFrozenLakeWrapper

        def make_env(**kwargs):
            change_notification = kwargs.get("change_notification", True)
            delta_change_notification = kwargs.get(
                "delta_change_notification", True,
            )

            base = gym.make(
                "FrozenLake-v1", is_slippery=False, max_episode_steps=50,
            )
            # fl_user_scheduler_cls / fl_user_update_cls are injected
            # into this exec sandbox by the cell below.
            sched = fl_user_scheduler_cls()
            upd = fl_user_update_cls(sched, T=20)
            return NSFrozenLakeWrapper(
                base, {"P": upd},
                change_notification=change_notification,
                delta_change_notification=delta_change_notification,
                initial_prob_dist=[1.0, 0.0, 0.0],
            )

        ENV_ID = "MyNSFrozenLake-v0"
        register(
            id=ENV_ID,
            entry_point=make_env,
            disable_env_checker=True,
            order_enforce=False,
        )
        ''')
    fl_register_editor = mo.ui.code_editor(
        value=_stub, language="python", min_height=22,
    )
    _ref = mo.ui.code_editor(
        value=_solution, language="python", disabled=True, min_height=22,
    )
    mo.ui.tabs({
        "✏️ Your code": fl_register_editor,
        "💡 Reference solution": _ref,
    })
    return (fl_register_editor,)


@app.cell
def _(fl_register_editor, fl_user_scheduler_cls, fl_user_update_cls):
    """Execute the user's registration block.

    We inject ``fl_user_scheduler_cls`` and ``fl_user_update_cls`` so
    the reference solution (and any user code that wants them) can
    refer to the compiled classes directly. The registered env id is
    surfaced as ``fl_register_env_id`` for the verifier below.
    """
    _ns = {
        "fl_user_scheduler_cls": fl_user_scheduler_cls,
        "fl_user_update_cls": fl_user_update_cls,
    }
    try:
        exec(fl_register_editor.value, _ns)
        fl_register_env_id = _ns.get("ENV_ID")
        fl_register_make_env = _ns.get("make_env")
        if fl_register_env_id is None:
            raise NameError("set ENV_ID = \"YourId-v0\" in your code")
        if fl_register_make_env is None:
            raise NameError("define a function named make_env")
        fl_register_ok = True
        _msg = (
            f"✓ registered `{fl_register_env_id}` — "
            f"call `gym.make(\"{fl_register_env_id}\")` anywhere."
        )
    except Exception as _e:
        fl_register_env_id = None
        fl_register_make_env = None
        fl_register_ok = False
        _msg = f"⚠ {type(_e).__name__}: {_e}"
    _msg
    return fl_register_env_id, fl_register_make_env, fl_register_ok


@app.cell
def _(fl_register_env_id, fl_register_ok, gym, mo):
    """Verify the registration by loading the env via gym.make and
    rolling a few random steps. Reacts live to edits in the editor
    above."""
    if not fl_register_ok or fl_register_env_id is None:
        _out = mo.md(
            "⚠ Fix the registration block above to see the verifier."
        )
    else:
        try:
            _env = gym.make(fl_register_env_id)
            _obs, _ = _env.reset(seed=0)
            _state = (
                _obs["state"] if isinstance(_obs, dict) else _obs
            )
            _slip = (
                _env.transition_prob
                if hasattr(_env, "transition_prob")
                else _env.unwrapped.transition_prob
                if hasattr(_env.unwrapped, "transition_prob") else None
            )
            _changes = 0
            for _t in range(10):
                _a = int(_env.action_space.sample())
                _o, _r, _d, _tr, _info = _env.step(_a)
                if isinstance(_o, dict):
                    _ec = _o.get("env_change", {})
                    _changes += int(any(v for v in _ec.values()))
                if _d or _tr:
                    break
            _out = mo.md(
                f"✓ `gym.make({fl_register_env_id!r})` succeeded.\n\n"
                f"- env: `{type(_env).__name__}` wrapping "
                f"`{type(_env.unwrapped).__name__}`\n"
                f"- initial state: `{_state}`\n"
                f"- initial slip distribution: `{_slip}`\n"
                f"- 10 random steps fired `{_changes}` env-change "
                "notifications\n\n"
                f"_Anyone can now load this env with_ "
                f"`gym.make({fl_register_env_id!r})` "
                "_— the same pattern the AAMAS 2026 competition_ "
                "_evaluator uses._"
            )
        except Exception as _e:
            _out = mo.md(
                f"⚠ `gym.make({fl_register_env_id!r})` failed: "
                f"`{type(_e).__name__}: {_e}`."
            )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Demo — full pipeline with your saved code</div></div>

    ### Builds a FrozenLake env using **your** `FLUserScheduler` + `FLUserUpdate`, then evaluates the **stale / replan / oracle** trio on it (10 seeds). End-to-end smoke test that your code wires into a real NS-MDP — and the result reacts live to any edits you make above.
    """)
    return


@app.cell
def _(
    NSFrozenLakeWrapper,
    fl_oracle_gamma,
    fl_oracle_horizon,
    fl_oracle_rollout,
    fl_replan_rollout,
    fl_rollout,
    fl_stationary_policy,
    fl_user_scheduler_cls,
    fl_user_update_cls,
    fl_vi_policy,
    gym,
    mo,
    np,
    ns_oracle_vi,
):
    if fl_user_scheduler_cls is None or fl_user_update_cls is None:
        _out = mo.md(
            "⚠ Compile both `FLUserScheduler` and `FLUserUpdate` above "
            "(look for the two ✓ messages) to see this demo."
        )
    else:
        try:
            def _make_user_env():
                base = gym.make(
                    "FrozenLake-v1", is_slippery=False,
                    max_episode_steps=50,
                )
                sched = fl_user_scheduler_cls()
                upd = fl_user_update_cls(sched)
                return NSFrozenLakeWrapper(
                    base, {"P": upd},
                    change_notification=True,
                    delta_change_notification=True,
                    initial_prob_dist=[1.0, 0.0, 0.0],
                )

            # Build a one-shot oracle (P, R) tensor for the user's env.
            _T = fl_oracle_horizon
            _probe = _make_user_env()
            _probe.reset(seed=0)
            _S = _probe.unwrapped.observation_space.n
            _A = _probe.unwrapped.action_space.n
            _P = np.zeros((_T, _S, _A, _S), dtype=float)
            _R = np.zeros((_T, _S, _A, _S), dtype=float)
            _env_tensor = _make_user_env()
            _env_tensor.reset(seed=0)
            for _t in range(_T):
                _env_tensor.transition_prob, _fired, _ = (
                    _env_tensor.update_fn(_env_tensor.transition_prob, _t)
                )
                if _fired:
                    _env_tensor._update_transition_prob_table()
                for _s in range(_S):
                    for _a in range(_A):
                        for _prob, _sp, _rew, _ in _env_tensor.P[_s][_a]:
                            _P[_t, _s, _a, _sp] += float(_prob)
                            _R[_t, _s, _a, _sp] = float(_rew)
            _, _oracle_pi = ns_oracle_vi(_P, _R, fl_oracle_gamma)

            _seeds = list(range(10))
            _stale, _replan, _oracle = [], [], []
            for _seed in _seeds:
                np.random.seed(_seed)
                _e1 = _make_user_env()
                _ra, _, _ = fl_rollout(_e1, fl_vi_policy, seed=_seed)
                np.random.seed(_seed)
                _e2 = _make_user_env()
                _rb, _ = fl_replan_rollout(
                    _e2, fl_stationary_policy, seed=_seed,
                )
                np.random.seed(_seed)
                _e3 = _make_user_env()
                _rc = fl_oracle_rollout(_e3, _oracle_pi, seed=_seed)
                _stale.append(float(_ra))
                _replan.append(float(_rb))
                _oracle.append(float(_rc))

            _out = mo.md(
                "**Result — 10 seeds on your env "
                "(`FLUserScheduler` × `FLUserUpdate`):**\n\n"
                f"- stale-VI  : μ = **{np.mean(_stale):.2f}**  "
                f"({sum(1 for r in _stale if r > 0)}/10 succ)\n"
                f"- replan-VI : μ = **{np.mean(_replan):.2f}**  "
                f"({sum(1 for r in _replan if r > 0)}/10 succ)\n"
                f"- oracle-VI : μ = **{np.mean(_oracle):.2f}**  "
                f"({sum(1 for r in _oracle if r > 0)}/10 succ)\n\n"
                "_Edit the scheduler or update editors above and this "
                "cell re-runs automatically._"
            )
        except Exception as _e:
            _out = mo.md(
                f"⚠ pipeline failed: `{type(_e).__name__}: {_e}`. "
                "Most likely your `_update` signature or return shape "
                "is off — `param` is a 3-list, return must also be a "
                "3-list."
            )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Your NS-MDP — combine the two galleries</div></div>

    ### The dropdowns above pick a scheduler (*when*) and an update function (*how*) independently. Below we **combine your selections** into a `your_nsmdp` FrozenLake env and run the same three planners from Section 3 against it: **stale VI** (no updates), **myopic replan VI** (re-solves stationarily on the live `P_t`), and **oracle VI** (time-augmented backward induction with full schedule foresight).

    ### Both the side-by-side heatmap and the per-episode return plot update reactively whenever you change a dropdown.

    > - **V^π under truth** — backward-pass policy evaluation of each
    >   planner's policy on the *true* non-stationary tensor (no max,
    >   just plug in π).
    > - **Action gap** — `max_a Q*(s,a,t) − second_max_a Q*`. Where
    >   the gap is large the optimal action is decisive (picking
    >   wrong costs a lot);
    """)
    return


@app.cell
def _(
    NSFrozenLakeWrapper,
    fl_init_intended,
    fl_make_gallery_scheduler,
    fl_make_gallery_update,
    gym,
):
    def fl_your_make_env():
        """Build a fresh wrapped FrozenLake using the two gallery selections.

        Each call constructs a fresh scheduler + update_fn so per-instance
        state (RNGs, cycle indices, total-change accumulators) is clean.
        """
        env = gym.make(
            "FrozenLake-v1", is_slippery=False, max_episode_steps=50,
        )
        sched = fl_make_gallery_scheduler()
        update_fn = fl_make_gallery_update(sched)
        return NSFrozenLakeWrapper(
            env,
            {"P": update_fn},
            change_notification=True,
            delta_change_notification=True,
            initial_prob_dist=init_dist(fl_init_intended.value),
        )

    return (fl_your_make_env,)


@app.cell
def _(build_ns_tensors, fl_oracle_horizon, fl_your_make_env):
    fl_your_P, fl_your_R = build_ns_tensors(
        fl_your_make_env, fl_oracle_horizon, seed=0,
    )
    fl_your_P.shape, fl_your_R.shape
    return fl_your_P, fl_your_R


@app.cell
def _(fl_oracle_gamma, fl_your_P, fl_your_R, ns_oracle_vi):
    fl_your_oracle_V, fl_your_oracle_policy = ns_oracle_vi(
        fl_your_P, fl_your_R, fl_oracle_gamma,
    )
    return (fl_your_oracle_policy,)


@app.cell
def _(
    fl_oracle_gamma,
    fl_stationary_V,
    fl_stationary_policy,
    fl_your_P,
    fl_your_R,
    np,
    stationary_vi_jit,
):
    """Myopic-replan agent's plan at each decision time.

    At decision time t the agent has observed the env state *after*
    step t-1 — i.e., the table that was in force during the (t-1)-th
    transition (= ``fl_your_P[t-1]`` after the build_ns_tensors fix).
    For t=0 the agent has taken no steps yet and uses the initial
    (stale) plan — matching what ``fl_replan_rollout`` actually passes
    as ``initial_policy``. So at t=0, myopic ≡ stale by construction.
    """
    _T = fl_your_P.shape[0]
    _S = fl_your_P.shape[1]
    fl_your_myopic_V = np.zeros((_T, _S))
    fl_your_myopic_policy = np.zeros((_T, _S), dtype=np.int64)
    fl_your_myopic_V[0] = fl_stationary_V
    fl_your_myopic_policy[0] = fl_stationary_policy.astype(np.int64)
    for _t in range(1, _T):
        _v, _pi = stationary_vi_jit(
            fl_your_P[_t - 1], fl_your_R[_t - 1], fl_oracle_gamma,
        )
        fl_your_myopic_V[_t] = _v
        fl_your_myopic_policy[_t] = _pi
    return (fl_your_myopic_policy,)


@app.cell
def _(
    fl_oracle_gamma,
    fl_stationary_policy,
    fl_your_P,
    fl_your_R,
    fl_your_myopic_policy,
    fl_your_oracle_policy,
    np,
    oracle_q_gap_jit,
    policy_eval_ns_jit,
):
    """Apples-to-apples evaluation: each planner's policy under truth.

    V^π_*(s, t) is the expected return from (s, t) when actions follow
    the planner's policy AND dynamics follow the true (P_t, R_t). Unlike
    V*, all three are on the same horizon and same env, so panel-to-panel
    differences reflect policy quality alone.
    """
    _T = fl_your_P.shape[0]
    _S = fl_your_P.shape[1]
    # Stale policy is constant in t — broadcast (S,) → (T, S).
    _stale_pi_tt = np.broadcast_to(
        fl_stationary_policy.astype(np.int64), (_T, _S)
    ).copy()
    fl_your_stale_Vpi = policy_eval_ns_jit(
        fl_your_P, fl_your_R, fl_oracle_gamma, _stale_pi_tt,
    )
    fl_your_myopic_Vpi = policy_eval_ns_jit(
        fl_your_P, fl_your_R, fl_oracle_gamma,
        fl_your_myopic_policy.astype(np.int64),
    )
    fl_your_oracle_Vpi = policy_eval_ns_jit(
        fl_your_P, fl_your_R, fl_oracle_gamma,
        fl_your_oracle_policy.astype(np.int64),
    )
    # Oracle Q* and action gap (max − second-best Q*) at each (s, t).
    fl_your_oracle_Q, fl_your_oracle_gap = oracle_q_gap_jit(
        fl_your_P, fl_your_R, fl_oracle_gamma,
    )
    return (
        fl_your_myopic_Vpi,
        fl_your_oracle_Vpi,
        fl_your_oracle_gap,
        fl_your_stale_Vpi,
    )


@app.cell
def _(fl_oracle_horizon, mo):
    fl_your_t = mo.ui.slider(
        start=0, stop=fl_oracle_horizon - 1, step=1, value=0,
        label="your-NS-MDP time t",
    )
    fl_your_view = mo.ui.dropdown(
        options=[
            "V^π  —  policy evaluated under truth",
            "Action gap  —  max Q* − second-best Q* (oracle, shared scale)",
        ],
        value="V^π  —  policy evaluated under truth",
        label="heatmap mode",
    )
    mo.hstack([fl_your_t, fl_your_view], justify="start", gap=2)
    return fl_your_t, fl_your_view


@app.cell
def _(
    FL_ACTION_LEGEND,
    fl_draw_policy_panel,
    fl_draw_value_panel,
    fl_drift_pick,
    fl_sched_pick,
    fl_stationary_policy,
    fl_your_myopic_Vpi,
    fl_your_myopic_policy,
    fl_your_oracle_Vpi,
    fl_your_oracle_gap,
    fl_your_oracle_policy,
    fl_your_stale_Vpi,
    fl_your_t,
    fl_your_view,
    plt,
):
    _t = fl_your_t.value
    _mode = fl_your_view.value

    # Pick the heatmap data + label based on the mode. All three planners
    # share a common color scale per mode for fair side-by-side reading.
    if _mode.startswith("Action gap"):
        # Oracle action gap (max Q* − second-best Q*) — single source of
        # truth, shared across all three panels with each planner's arrows.
        _gap = fl_your_oracle_gap[_t].reshape(4, 4)
        _grid_stale = _gap
        _grid_myopic = _gap
        _grid_oracle = _gap
        _vmax = max(float(fl_your_oracle_gap.max()), 1e-3)
        _row_label = f"Action gap (oracle Q*) at t={_t}"
        _t_stale_v = f"Stale  arrows on gap (t={_t})"
        _t_myopic_v = f"Myopic  arrows on gap (t={_t})"
        _t_oracle_v = f"Oracle  arrows on gap (t={_t})"
    else:
        # Default: V^π under truth (backward-pass policy evaluation on the
        # true (P_t, R_t) — apples-to-apples across planners).
        _grid_stale = fl_your_stale_Vpi[_t].reshape(4, 4)
        _grid_myopic = fl_your_myopic_Vpi[_t].reshape(4, 4)
        _grid_oracle = fl_your_oracle_Vpi[_t].reshape(4, 4)
        _vmax = max(
            float(fl_your_stale_Vpi[0].max()),
            float(fl_your_myopic_Vpi[0].max()),
            float(fl_your_oracle_Vpi[0].max()),
            1e-3,
        )
        _row_label = f"V^π(s, t={_t})  under truth"
        _t_stale_v = f"Stale  V^π(s, t={_t})"
        _t_myopic_v = f"Myopic replan  V^π(s, t={_t})"
        _t_oracle_v = f"Oracle  V^π(s, t={_t})"

    _fig, _axes = plt.subplots(2, 3, figsize=(13.5, 8.6), layout="constrained")
    # Top row — heatmap (with arrows). Arrows are always the planner's policy.
    fl_draw_value_panel(
        _axes[0, 0],
        _grid_stale,
        fl_stationary_policy.reshape(4, 4),
        _t_stale_v,
        vmax=_vmax,
    )
    fl_draw_value_panel(
        _axes[0, 1],
        _grid_myopic,
        fl_your_myopic_policy[_t].reshape(4, 4),
        _t_myopic_v,
        vmax=_vmax,
    )
    _im_v = fl_draw_value_panel(
        _axes[0, 2],
        _grid_oracle,
        fl_your_oracle_policy[_t].reshape(4, 4),
        _t_oracle_v,
        vmax=_vmax,
    )
    # Bottom row — π(s, t) color-coded action heatmaps
    fl_draw_policy_panel(
        _axes[1, 0],
        fl_stationary_policy.reshape(4, 4),
        "π(s) — stale",
    )
    fl_draw_policy_panel(
        _axes[1, 1],
        fl_your_myopic_policy[_t].reshape(4, 4),
        f"π(s, t={_t}) — myopic",
    )
    fl_draw_policy_panel(
        _axes[1, 2],
        fl_your_oracle_policy[_t].reshape(4, 4),
        f"π(s, t={_t}) — oracle",
    )

    _short_sched = fl_sched_pick.value.split(" — ")[0].split("(")[0].strip()
    _short_drift = fl_drift_pick.value.split(" — ")[0].split("(")[0].strip()
    _fig.suptitle(
        f"YOUR env  •  {_short_sched}  ×  {_short_drift}  •  "
        f"top: {_row_label}   bottom: π(s,t)",
        fontsize=10,
    )
    _fig.colorbar(_im_v, ax=_axes[0, :], fraction=0.025, pad=0.02)
    _fig.legend(
        handles=FL_ACTION_LEGEND,
        loc="lower center", ncol=4, fontsize=9,
        bbox_to_anchor=(0.5, -0.01), frameon=False,
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #555; background: #f6f6f6;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Performance — stale vs replan vs oracle on your env</div></div>

    ### **100 episodes** per planner, same drift schedule, fresh RNGs per episode.

    ### - **Left — xreturn distribution per planner.**
    ### - Violin shape carries the median, IQR and the long tail of zeros. The dot marks the mean; the annotation gives success rate over all seeds.
    ###- **Right — paired difference.** For each seed, *(oracle − stale)* and *(oracle − replan)*. Dots above 0 mean oracle wins on that seed; dots below 0 are the surprising losses you noticed. The vertical band is the mean ± SE of the per-seed difference.
    """)
    return


@app.cell
def _(
    fl_drift_pick,
    fl_oracle_policy,
    fl_oracle_rollout,
    fl_replan_rollout,
    fl_rollout,
    fl_sched_pick,
    fl_stationary_policy,
    fl_vi_policy,
    fl_your_make_env,
    fl_your_oracle_policy,
    mo,
    np,
    plt,
):
    _ = fl_oracle_policy  # silence unused import; we use fl_your_oracle_policy
    _N_SEEDS = 100
    _seeds = list(range(_N_SEEDS))
    _stale, _replan, _oracle = [], [], []
    for _s in mo.status.progress_bar(
        _seeds, title="Stale vs replan vs oracle on your env",
        subtitle=f"rolling out 3 planners × {_N_SEEDS} seeds…",
        remove_on_exit=True,
    ):
        _env_a = fl_your_make_env()
        _env_b = fl_your_make_env()
        _env_c = fl_your_make_env()
        np.random.seed(_s)
        _ra, _, _ = fl_rollout(_env_a, fl_vi_policy, seed=_s)
        np.random.seed(_s)
        _rb, _ = fl_replan_rollout(_env_b, fl_stationary_policy, seed=_s)
        np.random.seed(_s)
        _rc = fl_oracle_rollout(_env_c, fl_your_oracle_policy, seed=_s)
        _stale.append(float(_ra))
        _replan.append(float(_rb))
        _oracle.append(float(_rc))
    _stale_a = np.asarray(_stale)
    _replan_a = np.asarray(_replan)
    _oracle_a = np.asarray(_oracle)

    _fig, (_ax_v, _ax_d) = plt.subplots(
        1, 2, figsize=(11.5, 3.8), layout="constrained",
    )

    # --- Left: violin plot of return distributions ----------------
    _data = [_stale_a, _replan_a, _oracle_a]
    _names = ["stale-VI", "replan-VI", "oracle-VI"]
    _colors = ["tab:red", "tab:blue", "tab:green"]
    _vp = _ax_v.violinplot(
        _data, positions=[0, 1, 2], showmeans=False,
        showmedians=False, showextrema=False, widths=0.75,
    )
    for _body, _col in zip(_vp["bodies"], _colors):
        _body.set_facecolor(_col)
        _body.set_edgecolor("black")
        _body.set_alpha(0.55)
        _body.set_linewidth(0.6)
    # Mean dot + SE bar for each violin.
    for _i, (_arr, _col) in enumerate(zip(_data, _colors)):
        _mu = float(np.mean(_arr))
        _se = float(np.std(_arr, ddof=1) / np.sqrt(len(_arr)))
        _ax_v.errorbar(
            _i, _mu, yerr=_se, fmt="o", color="white",
            markerfacecolor=_col, markeredgecolor="black",
            markersize=8, capsize=4, zorder=5,
        )
        _succ = int(np.sum(_arr > 0))
        _ax_v.text(
            _i, 1.08,
            f"μ={_mu:.2f}±{_se:.02f}\n{_succ}/{_N_SEEDS} succ",
            ha="center", va="bottom", fontsize=8,
        )
    _ax_v.set_xticks([0, 1, 2])
    _ax_v.set_xticklabels(_names, fontsize=9)
    _ax_v.set_ylabel("return  (1 = goal reached)")
    _ax_v.set_ylim(-0.1, 1.3)
    _ax_v.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
    _ax_v.axhline(1, color="grey", linewidth=0.5, alpha=0.4)
    _ax_v.set_title(
        f"Return distribution  (violins, N={_N_SEEDS})", fontsize=10,
    )

    # --- Right: paired-diff strip plot ---------------------------
    _diff_oracle_stale = _oracle_a - _stale_a
    _diff_oracle_replan = _oracle_a - _replan_a
    _diffs = [_diff_oracle_stale, _diff_oracle_replan]
    _diff_names = ["oracle − stale", "oracle − replan"]
    _diff_colors = ["tab:red", "tab:blue"]  # color of the *baseline*
    _rng = np.random.default_rng(0)
    for _i, (_d, _col) in enumerate(zip(_diffs, _diff_colors)):
        _xs = _i + (_rng.uniform(-0.16, 0.16, size=len(_d)))
        _ax_d.scatter(
            _xs, _d, s=18, alpha=0.55, color=_col,
            edgecolor="black", linewidth=0.3,
        )
        _mu = float(np.mean(_d))
        _se = float(np.std(_d, ddof=1) / np.sqrt(len(_d)))
        _wins = int(np.sum(_d > 0))
        _losses = int(np.sum(_d < 0))
        _ax_d.errorbar(
            _i, _mu, yerr=_se, fmt="D", color="white",
            markerfacecolor="black", markeredgecolor="black",
            markersize=7, capsize=5, zorder=5,
        )
        _ax_d.text(
            _i, 1.18,
            f"μ_diff={_mu:+.2f}±{_se:.02f}\n"
            f"oracle wins {_wins}/{_N_SEEDS}  ·  "
            f"loses {_losses}/{_N_SEEDS}",
            ha="center", va="bottom", fontsize=8,
        )
    _ax_d.axhline(0, color="grey", linewidth=0.8)
    _ax_d.set_xticks([0, 1])
    _ax_d.set_xticklabels(_diff_names, fontsize=9)
    _ax_d.set_ylabel("per-seed return diff")
    _ax_d.set_ylim(-1.4, 1.4)
    _ax_d.set_title(
        "Paired difference per seed  (above 0 → oracle wins)",
        fontsize=10,
    )

    _short_sched = fl_sched_pick.value.split(" — ")[0].split("(")[0].strip()
    _short_drift = fl_drift_pick.value.split(" — ")[0].split("(")[0].strip()
    _fig.suptitle(
        f"YOUR env  •  {_short_sched}  ×  {_short_drift}  •  "
        f"N={_N_SEEDS}",
        fontsize=10,
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #5a3a8a; background: #f3eef7;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">The three notification levels</div></div>

    1. **Basic** — `change_notification=True`. Boolean: did `P` change?
    2. **Detailed** — also `delta_change_notification=True`. Magnitude of
       ### the change (Wasserstein distance for distributions, scalar delta for scalar params).
    3. **Full env model** — call `ns_env.get_planning_env()` for a stationary
       ### snapshot of the current MDP. With detailed notifications on, the snapshot reflects up-to-date parameters; without them you get a snapshot of the base env. Used by planning algorithms like MCTS.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    fl_run_sweep = mo.ui.run_button(label="Run FrozenLake sweep")
    mo.vstack([
        mo.md(r"""
        <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Sweep — random-policy returns vs drift</div></div>

        ### Vary `k` from 0 to 0.3, run 30 episodes per setting with a uniformly random policy. Result is stashed in scope as `fl_sweep` for Section 4 and saved to `results/nb_1_frozenlake.json` for offline analysis.
        """),
        fl_run_sweep,
    ])
    return (fl_run_sweep,)


@app.cell
def _(
    fl_make_env,
    fl_random_policy,
    fl_rollout,
    fl_run_sweep,
    mo,
    np,
    save_sweep,
):
    import time as _time
    fl_sweep = None
    fl_sweep_path = None
    if not fl_run_sweep.value:
        _status = mo.md(
            "_Click **Run FrozenLake sweep** above to run the "
            "random-policy baseline (~10 s)._"
        )
    else:
        _t0 = _time.time()
        _ks = np.linspace(0.0, 0.3, 7)
        _returns_by_k = {}
        for _k in mo.status.progress_bar(
            _ks, title="FrozenLake random sweep", subtitle="rolling out…",
            remove_on_exit=True,
        ):
            _returns = []
            for _ep in range(30):
                _env_k = fl_make_env(_k)
                _ret, _, _ = fl_rollout(_env_k, fl_random_policy, seed=_ep)
                _returns.append(float(_ret))
            _returns_by_k[float(_k)] = _returns
        fl_sweep = {
            "env": "FrozenLake-v1",
            "param": "P",
            "scheduler": "ContinuousScheduler",
            "update_fn": "DistributionDecrementUpdate",
            "policy": "random",
            "n_episodes": 30,
            "returns_by_k": _returns_by_k,
        }
        fl_sweep_path = save_sweep("nb_1_frozenlake.json", fl_sweep)
        _status = mo.md(
            f"✓ Sweep finished in **{_time.time() - _t0:.1f} s** — "
            f"7 drift levels × 30 episodes. Saved to `{fl_sweep_path}`."
        )
    _status
    return (fl_sweep,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #5a3a8a; background: #f3eef7;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Bridge — the canonical two-path NS-MDP</div></div>

    ### FrozenLake's geometry forces all 6-step paths to be roughly equally risky. **Bridge** (Lecarpentier & Rachelson, NeurIPS 2019) was designed to break that symmetry — and it's the env every NS-MDP paper since RATS uses as the headline benchmark.

    ```
    H H H H H H H H        row 0
    F F F F F H H H        row 1
    G F F F S F F G        row 2   ← S=start, G=goals on both ends
    F F F F F H H H        row 3
    H H H H H H H H        row 4
    ```

    ### Two routes from `S`:

    - **Right (3 steps)** — shorter but every perpendicular slip lands
      ### in a hole (rows 1/3 cols 5–7 are all `H`).
    - **Left (4 steps)** — one step longer, but every perpendicular
      ### slip lands on a frozen tile (rows 1/3 cols 0–4 are all `F`).

    ### Under deterministic dynamics both reach the goal; right is faster. Under stochastic slip the right bridge becomes deadly. **Oracle**, knowing the future drift schedule, commits left from `t=0`. **Myopic** at `t=0` sees a deterministic `P_0` and commits right — by the time it can replan, it's already on the narrow bridge.

    ### The same gallery dropdowns drive the env: `fl_sched_pick` and `fl_drift_pick` control *when* and *how* the slip evolves on Bridge too. (Bridge has its own per-side params `left_side` / `right_side` too; we use the global `P` here for parity with FrozenLake.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bridge_init_left = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=1.0,
        label="Bridge left — initial P[intended]",
    )
    bridge_init_right = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=1.0,
        label="Bridge right — initial P[intended]",
    )
    mo.vstack([
        mo.md(r"""
        <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #555; background: #f6f6f6;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Bridge — initial transition probabilities</div></div>

        ### Set the **initial** `P[intended]` for each half of the map. In split-prob mode, left and right bridges take different starting distributions; the right corridor's initial value matters because the agent can't avoid stepping onto holes early if you crank it down.
        """),
        bridge_init_left,
        bridge_init_right,
    ])
    return bridge_init_left, bridge_init_right


@app.cell
def _():
    BRIDGE_MAP = ["HHHHHHHH",
                  "FFFFFHHH", 
                  "GFHFSFFG", 
                  "FFFFFHHH", 
                  "HHHHHHHH"]
    BRIDGE_NROW = 5
    BRIDGE_NCOL = 8
    BRIDGE_NSTATE = BRIDGE_NROW * BRIDGE_NCOL  # 40
    BRIDGE_NACTION = 4
    return BRIDGE_MAP, BRIDGE_NCOL, BRIDGE_NROW


@app.cell
def _(np):
    def bridge_value_iteration(env, gamma=0.95, theta=1e-6, max_iter=2000):
        """Full infinite-horizon stationary VI on a Bridge env.

        Bridge stores `env.unwrapped.P` as a 3-element categorical
        `[p_intended, p_perp1, p_perp2]`, NOT a gym-style
        `{s: {a: [(prob, sp, r, done), ...]}}` dict. We call
        `_get_transition_matrix()` to build the gym-style dict from
        the categorical + map geometry, then run standard VI.
        """
        T_dict = env.unwrapped._get_transition_matrix()
        n_states = env.unwrapped.observation_space.n
        n_actions = env.unwrapped.action_space.n
        V = np.zeros(n_states)
        for _ in range(max_iter):
            delta = 0.0
            for s in range(n_states):
                v_old = V[s]
                V[s] = max(
                    sum(prob * (reward + gamma * V[next_state])
                        for prob, next_state, reward, _ in T_dict[s][a])
                    for a in range(n_actions)
                )
                delta = max(delta, abs(v_old - V[s]))
            if delta < theta:
                break
        policy = np.zeros(n_states, dtype=np.int64)
        for s in range(n_states):
            policy[s] = int(np.argmax([
                sum(prob * (reward + gamma * V[next_state])
                    for prob, next_state, reward, _ in T_dict[s][a])
                for a in range(n_actions)
            ]))
        return policy, V

    return (bridge_value_iteration,)


@app.cell
def _(NSBridgeWrapper, bridge_value_iteration, gym):
    # Stale baseline: VI on deterministic Bridge (initial_prob_dist=[1,0,0]).
    _stale_env = NSBridgeWrapper(
        gym.make("ns_gym/Bridge-v0", max_episode_steps=50),
        {},  # no NS update — pure stationary
        change_notification=False,
        delta_change_notification=False,
        initial_prob_dist=[1.0, 0.0, 0.0],
    )
    _stale_env.reset(seed=0)
    bridge_stationary_policy, bridge_stationary_V = bridge_value_iteration(_stale_env)
    return bridge_stationary_V, bridge_stationary_policy


@app.cell(hide_code=True)
def _(DRIFT_OPTIONS, mo):
    bridge_split_sides = mo.ui.checkbox(
        value=True,
        label="🌉 Drift left and right sides independently (split-prob mode)",
    )
    bridge_left_drift = mo.ui.dropdown(
        options=DRIFT_OPTIONS,
        value="DistributionDecrementUpdate(k=0.05)",
        label="left-side drift  (cols 0–4 of the bridge)",
    )
    bridge_right_drift = mo.ui.dropdown(
        options=DRIFT_OPTIONS,
        value="★ SigmoidEarlyCliff(t0=4, k=1.0) — oracle ≠ myopic",
        label="right-side drift  (cols 4–7, holes flank the corridor)",
    )
    mo.vstack([
        mo.md(r"""
        <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #555; background: #f6f6f6;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Bridge — per-side drift control</div></div>

        ### Bridge ships with three tunable params: a global `P`, plus independent `left_side` and `right_side` categorical distributions. With the **split** mode below, the wrapper applies different update functions to the left and right halves of the map. Default recipe makes the right (narrow) corridor increasingly slippery while the left (safe) bridge stays calm — oracle commits left from t=0, myopic gambles right and dies.

        ### Uncheck the box to fall back to global-P drift (uses the main `fl_drift_pick` selection above).

        > **★ Heatmap modes (same as the FrozenLake panel).** Use the
        > **heatmap mode** dropdown to pick between:
        >
        > - **V^π under truth** *(default)* — backward-pass policy
        >   evaluation of each planner's policy on the *true*
        >   non-stationary tensor (no max, just plug in π). All three
        >   are expected return under the same dynamics over the same
        >   horizon — apples-to-apples; the panel-to-panel gap is
        >   exactly the policy-quality gap the bar plot below measures.
        > - **Action gap** — `max_a Q*(s,a,t) − second_max_a Q*`.
        >   Where the gap is large the optimal action is decisive
        >   (picking wrong costs a lot); where it's small, multiple
        >   actions are roughly tied. Invariant to horizon scaling.
        >   The colormap is shared across all three panels (same gap
        >   field); the arrows on each panel are that planner's policy,
        >   so you can spot disagreements on cells where the gap
        >   actually matters.
        """),
        bridge_split_sides,
        bridge_left_drift,
        bridge_right_drift,
    ])
    return bridge_left_drift, bridge_right_drift, bridge_split_sides


@app.cell
def _(
    NSBridgeWrapper,
    bridge_init_left,
    bridge_init_right,
    bridge_left_drift,
    bridge_right_drift,
    bridge_split_sides,
    fl_make_gallery_scheduler,
    fl_make_gallery_update,
    gym,
):
    def bridge_make_env():
        """Build a Bridge env from the live gallery selections.

        If `bridge_split_sides` is checked, drives left/right sides
        independently via the per-side drift dropdowns and per-side
        initial-prob sliders. Otherwise uses the global `fl_drift_pick`
        selection on `P` and the left-side init slider as the global
        initial distribution.
        """
        if bridge_split_sides.value:
            env = gym.make(
                "ns_gym/Bridge-v0", max_episode_steps=50, split_probs=True,
            )
            sched_left = fl_make_gallery_scheduler()
            sched_right = fl_make_gallery_scheduler()
            update_left = fl_make_gallery_update(
                sched_left, drift_value=bridge_left_drift.value,
            )
            update_right = fl_make_gallery_update(
                sched_right, drift_value=bridge_right_drift.value,
            )
            tunable_params = {
                "P_left": update_left,
                "P_right": update_right,
            }
        else:
            env = gym.make("ns_gym/Bridge-v0", max_episode_steps=50)
            sched = fl_make_gallery_scheduler()
            update_fn = fl_make_gallery_update(sched)
            tunable_params = {"P": update_fn}

        # The wrapper accepts a 2-tuple `initial_prob_dist=(left, right)`
        # for asymmetric init. This is the only way to make per-side
        # init survive `reset()` calls (e.g. the ones inside
        # build_bridge_tensors).
        if bridge_split_sides.value:
            init_pd = (
                init_dist(bridge_init_left.value),
                init_dist(bridge_init_right.value),
            )
        else:
            init_pd = init_dist(bridge_init_left.value)
        return NSBridgeWrapper(
            env,
            tunable_params,
            change_notification=True,
            delta_change_notification=True,
            initial_prob_dist=init_pd,
        )

    return (bridge_make_env,)


@app.cell
def _(np):
    def build_bridge_tensors(make_env_fn, T, seed=0):
        """Snapshot Bridge transition tensors at each of T timesteps.

        NSBridgeWrapper.step calls ``_step_update`` BEFORE the real
        transition, so the table that governs the agent's t-th step is
        the one after ``_step_update`` has run with ``self.t == t``. We
        replay this in-place on a single env (setting ``self.t = t``
        manually) and snapshot the resulting transition matrix.
        """
        env = make_env_fn()
        env.reset(seed=seed)
        n_states = env.unwrapped.observation_space.n
        n_actions = env.unwrapped.action_space.n
        P = np.zeros((T, n_states, n_actions, n_states), dtype=np.float64)
        R = np.zeros((T, n_states, n_actions, n_states), dtype=np.float64)
        for t in range(T):
            env.t = t
            env._step_update()
            T_dict = env.unwrapped._get_transition_matrix()
            for s in range(n_states):
                for a in range(n_actions):
                    for prob, sp, reward, _ in T_dict[s][a]:
                        P[t, s, a, sp] += float(prob)
                        R[t, s, a, sp] = float(reward)
        return P, R

    return (build_bridge_tensors,)


@app.cell
def _(bridge_make_env, build_bridge_tensors, fl_oracle_horizon):
    bridge_P, bridge_R = build_bridge_tensors(
        bridge_make_env, fl_oracle_horizon, seed=0,
    )
    bridge_P.shape, bridge_R.shape
    return bridge_P, bridge_R


@app.cell
def _(bridge_P, bridge_R, fl_oracle_gamma, ns_oracle_vi):
    bridge_oracle_V, bridge_oracle_policy = ns_oracle_vi(
        bridge_P, bridge_R, fl_oracle_gamma,
    )
    return bridge_oracle_V, bridge_oracle_policy


@app.cell
def _(
    bridge_P,
    bridge_R,
    bridge_stationary_V,
    bridge_stationary_policy,
    fl_oracle_gamma,
    np,
    stationary_vi_jit,
):
    """Myopic-replan agent's plan at each decision time on Bridge.

    At decision time t the agent has only observed the env state *after*
    step t-1 — i.e., the table that was in force during the (t-1)-th
    transition (= ``bridge_P[t-1]`` after the build_bridge_tensors fix).
    For t=0 the agent has taken no steps yet and uses the initial
    (stale) plan, matching what the rollouts pass as ``initial_policy``.
    So at t=0, myopic ≡ stale by construction.
    """
    _T = bridge_P.shape[0]
    _S = bridge_P.shape[1]
    bridge_myopic_V = np.zeros((_T, _S))
    bridge_myopic_policy = np.zeros((_T, _S), dtype=np.int64)
    bridge_myopic_V[0] = bridge_stationary_V
    bridge_myopic_policy[0] = bridge_stationary_policy.astype(np.int64)
    for _t in range(1, _T):
        _v, _pi = stationary_vi_jit(
            bridge_P[_t - 1], bridge_R[_t - 1], fl_oracle_gamma,
        )
        bridge_myopic_V[_t] = _v
        bridge_myopic_policy[_t] = _pi
    return (bridge_myopic_policy,)


@app.cell
def _(
    bridge_P,
    bridge_R,
    bridge_myopic_policy,
    bridge_oracle_policy,
    bridge_stationary_policy,
    fl_oracle_gamma,
    np,
    oracle_q_gap_jit,
    policy_eval_ns_jit,
):
    """Bridge — apples-to-apples evaluation under truth + oracle action gap."""
    _T = bridge_P.shape[0]
    _S = bridge_P.shape[1]
    _stale_pi_tt = np.broadcast_to(
        bridge_stationary_policy.astype(np.int64), (_T, _S)
    ).copy()
    bridge_stale_Vpi = policy_eval_ns_jit(
        bridge_P, bridge_R, fl_oracle_gamma, _stale_pi_tt,
    )
    bridge_myopic_Vpi = policy_eval_ns_jit(
        bridge_P, bridge_R, fl_oracle_gamma,
        bridge_myopic_policy.astype(np.int64),
    )
    bridge_oracle_Vpi = policy_eval_ns_jit(
        bridge_P, bridge_R, fl_oracle_gamma,
        bridge_oracle_policy.astype(np.int64),
    )
    bridge_oracle_Q, bridge_oracle_gap = oracle_q_gap_jit(
        bridge_P, bridge_R, fl_oracle_gamma,
    )
    return (
        bridge_myopic_Vpi,
        bridge_oracle_Vpi,
        bridge_oracle_gap,
        bridge_stale_Vpi,
    )


@app.cell
def _(fl_oracle_horizon, mo):
    bridge_t = mo.ui.slider(
        start=0, stop=fl_oracle_horizon - 1, step=1, value=0,
        label="Bridge time t",
    )
    bridge_view = mo.ui.dropdown(
        options=[
            "V^π  —  policy evaluated under truth",
            "Action gap  —  max Q* − second-best Q* (oracle, shared scale)",
        ],
        value="V^π  —  policy evaluated under truth",
        label="heatmap mode",
    )
    mo.hstack([bridge_t, bridge_view], justify="start", gap=2)
    return bridge_t, bridge_view


@app.cell
def _(
    BRIDGE_MAP,
    BRIDGE_NCOL,
    BRIDGE_NROW,
    FL_ACTION_ARROWS,
    FL_ACTION_COLORS,
    np,
):
    from matplotlib.colors import ListedColormap as _BridgeListedCmap

    _BRIDGE_ACTION_CMAP = _BridgeListedCmap(FL_ACTION_COLORS)
    _BRIDGE_ACTION_CMAP.set_bad(color="#dddddd")

    def bridge_draw_value_panel(ax, V_grid, policy_grid, title, vmax):
        """Render the Bridge V/π heatmap with overlays."""
        im = ax.imshow(V_grid, cmap="viridis", vmin=float(V_grid.min()), vmax=vmax)
        for r in range(BRIDGE_NROW):
            for c in range(BRIDGE_NCOL):
                tile = BRIDGE_MAP[r][c]
                if tile == "H":
                    ax.text(c, r, "H", ha="center", va="center",
                            color="red", fontsize=11, fontweight="bold")
                elif tile == "G":
                    ax.text(c, r, "G", ha="center", va="center",
                            color="gold", fontsize=11, fontweight="bold")
                else:
                    ax.text(c, r - 0.20,
                            FL_ACTION_ARROWS[int(policy_grid[r, c])],
                            ha="center", va="center",
                            color="white", fontsize=10)
                    ax.text(c, r + 0.30, f"{V_grid[r, c]:.2f}",
                            ha="center", va="center",
                            color="white", fontsize=6)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    def bridge_draw_policy_panel(ax, policy_grid, title):
        """Color-coded action heatmap for Bridge."""
        masked = np.array(policy_grid, dtype=float)
        for r in range(BRIDGE_NROW):
            for c in range(BRIDGE_NCOL):
                if BRIDGE_MAP[r][c] in ("H", "G"):
                    masked[r, c] = np.nan
        masked = np.ma.array(masked, mask=np.isnan(masked))
        im = ax.imshow(masked, cmap=_BRIDGE_ACTION_CMAP, vmin=-0.5, vmax=3.5)
        for r in range(BRIDGE_NROW):
            for c in range(BRIDGE_NCOL):
                tile = BRIDGE_MAP[r][c]
                if tile == "H":
                    ax.text(c, r, "H", ha="center", va="center",
                            color="red", fontsize=11, fontweight="bold")
                elif tile == "G":
                    ax.text(c, r, "G", ha="center", va="center",
                            color="gold", fontsize=11, fontweight="bold")
                else:
                    ax.text(c, r,
                            FL_ACTION_ARROWS[int(policy_grid[r, c])],
                            ha="center", va="center",
                            color="white", fontsize=12)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    return bridge_draw_policy_panel, bridge_draw_value_panel


@app.cell
def _(
    BRIDGE_NCOL,
    BRIDGE_NROW,
    FL_ACTION_LEGEND,
    bridge_draw_policy_panel,
    bridge_draw_value_panel,
    bridge_myopic_Vpi,
    bridge_myopic_policy,
    bridge_oracle_Vpi,
    bridge_oracle_gap,
    bridge_oracle_policy,
    bridge_stale_Vpi,
    bridge_stationary_policy,
    bridge_t,
    bridge_view,
    fl_drift_pick,
    fl_sched_pick,
    plt,
):
    _t = bridge_t.value
    _mode = bridge_view.value

    if _mode.startswith("Action gap"):
        _gap = bridge_oracle_gap[_t].reshape(BRIDGE_NROW, BRIDGE_NCOL)
        _grid_stale = _gap
        _grid_myopic = _gap
        _grid_oracle = _gap
        _vmax = max(float(bridge_oracle_gap.max()), 1e-3)
        _row_label = f"Action gap (oracle Q*) at t={_t}"
        _t_stale_v = f"Stale  arrows on gap (t={_t})"
        _t_myopic_v = f"Myopic  arrows on gap (t={_t})"
        _t_oracle_v = f"Oracle  arrows on gap (t={_t})"
    else:
        _grid_stale = bridge_stale_Vpi[_t].reshape(BRIDGE_NROW, BRIDGE_NCOL)
        _grid_myopic = bridge_myopic_Vpi[_t].reshape(BRIDGE_NROW, BRIDGE_NCOL)
        _grid_oracle = bridge_oracle_Vpi[_t].reshape(BRIDGE_NROW, BRIDGE_NCOL)
        _vmax = max(
            float(bridge_stale_Vpi[0].max()),
            float(bridge_myopic_Vpi[0].max()),
            float(bridge_oracle_Vpi[0].max()),
            1e-3,
        )
        _row_label = f"V^π(s, t={_t})  under truth"
        _t_stale_v = f"Stale  V^π(s, t={_t})"
        _t_myopic_v = f"Myopic replan  V^π(s, t={_t})"
        _t_oracle_v = f"Oracle  V^π(s, t={_t})"

    _fig, _axes = plt.subplots(
        2, 3, figsize=(16, 7),
        gridspec_kw={"height_ratios": [BRIDGE_NROW, BRIDGE_NROW]},
        layout="constrained",
    )
    # Top row — heatmap (with arrows). Arrows always show planner's policy.
    bridge_draw_value_panel(
        _axes[0, 0],
        _grid_stale,
        bridge_stationary_policy.reshape(BRIDGE_NROW, BRIDGE_NCOL),
        _t_stale_v,
        vmax=_vmax,
    )
    bridge_draw_value_panel(
        _axes[0, 1],
        _grid_myopic,
        bridge_myopic_policy[_t].reshape(BRIDGE_NROW, BRIDGE_NCOL),
        _t_myopic_v,
        vmax=_vmax,
    )
    _im_v = bridge_draw_value_panel(
        _axes[0, 2],
        _grid_oracle,
        bridge_oracle_policy[_t].reshape(BRIDGE_NROW, BRIDGE_NCOL),
        _t_oracle_v,
        vmax=_vmax,
    )
    # Bottom row — π(s, t)
    bridge_draw_policy_panel(
        _axes[1, 0],
        bridge_stationary_policy.reshape(BRIDGE_NROW, BRIDGE_NCOL),
        "π(s) — stale",
    )
    bridge_draw_policy_panel(
        _axes[1, 1],
        bridge_myopic_policy[_t].reshape(BRIDGE_NROW, BRIDGE_NCOL),
        f"π(s, t={_t}) — myopic",
    )
    bridge_draw_policy_panel(
        _axes[1, 2],
        bridge_oracle_policy[_t].reshape(BRIDGE_NROW, BRIDGE_NCOL),
        f"π(s, t={_t}) — oracle",
    )

    _short_sched = fl_sched_pick.value.split(" — ")[0].split("(")[0].strip()
    _short_drift = fl_drift_pick.value.split(" — ")[0].split("(")[0].strip()
    _fig.suptitle(
        f"BRIDGE  •  {_short_sched}  ×  {_short_drift}  •  "
        f"top: {_row_label}   bottom: π(s,t)",
        fontsize=10,
    )
    _fig.colorbar(_im_v, ax=_axes[0, :], fraction=0.025, pad=0.02)
    _fig.legend(
        handles=FL_ACTION_LEGEND,
        loc="lower center", ncol=4, fontsize=9,
        bbox_to_anchor=(0.5, -0.01), frameon=False,
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #555; background: #f6f6f6;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Performance — stale vs replan vs oracle on Bridge</div></div>

    ### **100 episodes** per planner, same drift schedule, fresh RNGs per episode. With the default recipie

    ### - **Left — return distribution per planner.** Violin shape carries the median, IQR, and the long tail of −1 hole-falls. The dot marks the mean ± SE; the annotation gives the goal-success rate. -

    ### - **Right — paired difference.** For each seed, *(oracle − stale)* and *(oracle − replan)*. Dots above 0 mean oracle wins on that seed; dots below 0 are the surprising losses. The black diamond is the mean ± SE of the per-seed difference.

    > **Variance note.** `Bridge.step()` samples slips from the *global*
    > numpy RNG, not the env-seeded one, so two rollouts with the same
    > `env.reset(seed=s)` can still see different transition realizations.
    > To compare planners apples-to-apples we reseed `np.random` before
    > each rollout — without this, replan can occasionally beat the
    > oracle on a single episode (the oracle is optimal *in expectation*,
    > not on every realization).
    """)
    return


@app.cell
def _(
    bridge_make_env,
    bridge_oracle_policy,
    bridge_stationary_policy,
    bridge_value_iteration,
    fl_drift_pick,
    fl_sched_pick,
    np,
    plt,
    type_mismatch_checker,
):
    def _bridge_rollout(env, policy_arr, seed):
        obs, _ = env.reset(seed=seed)
        state, _ = type_mismatch_checker(obs)
        done = trunc = False
        ep_return = 0.0
        while not (done or trunc):
            action = int(policy_arr[state])
            obs, reward, done, trunc, _ = env.step(action)
            state, reward = type_mismatch_checker(obs, reward)
            ep_return += reward
        return ep_return

    def _bridge_replan_rollout(env, initial_policy, seed):
        policy = np.asarray(initial_policy).copy()
        obs, _ = env.reset(seed=seed)
        state, _ = type_mismatch_checker(obs)
        done = trunc = False
        ep_return = 0.0
        while not (done or trunc):
            action = int(policy[state])
            obs, reward, done, trunc, _ = env.step(action)
            if isinstance(obs, dict) and obs.get("env_change", {}).get("P", 0):
                planning_env = env.get_planning_env()
                policy, _ = bridge_value_iteration(planning_env)
            state, reward = type_mismatch_checker(obs, reward)
            ep_return += reward
        return ep_return

    def _bridge_oracle_rollout(env, policy_st, seed):
        obs, _ = env.reset(seed=seed)
        state, _ = type_mismatch_checker(obs)
        done = trunc = False
        t = 0
        ep_return = 0.0
        T = policy_st.shape[0]
        while not (done or trunc):
            a = int(policy_st[min(t, T - 1), state])
            obs, reward, done, trunc, _ = env.step(a)
            state, reward = type_mismatch_checker(obs, reward)
            ep_return += reward
            t += 1
        return ep_return

    _N_SEEDS = 100
    _seeds = list(range(_N_SEEDS))
    _stale, _replan, _oracle = [], [], []
    for _s in _seeds:
        np.random.seed(_s)
        _stale.append(_bridge_rollout(bridge_make_env(), bridge_stationary_policy, seed=_s))
        np.random.seed(_s)
        _replan.append(_bridge_replan_rollout(bridge_make_env(), bridge_stationary_policy, seed=_s))
        np.random.seed(_s)
        _oracle.append(_bridge_oracle_rollout(bridge_make_env(), bridge_oracle_policy, seed=_s))
    _stale_a = np.asarray(_stale, dtype=float)
    _replan_a = np.asarray(_replan, dtype=float)
    _oracle_a = np.asarray(_oracle, dtype=float)

    _fig, (_ax_v, _ax_d) = plt.subplots(
        1, 2, figsize=(11.5, 3.8), layout="constrained",
    )

    # --- Left: violin plot of return distributions ----------------
    _data = [_stale_a, _replan_a, _oracle_a]
    _names = ["stale-VI", "replan-VI", "oracle-VI"]
    _colors = ["tab:red", "tab:blue", "tab:green"]
    _vp = _ax_v.violinplot(
        _data, positions=[0, 1, 2], showmeans=False,
        showmedians=False, showextrema=False, widths=0.75,
    )
    for _body, _col in zip(_vp["bodies"], _colors):
        _body.set_facecolor(_col)
        _body.set_edgecolor("black")
        _body.set_alpha(0.55)
        _body.set_linewidth(0.6)
    for _i, (_arr, _col) in enumerate(zip(_data, _colors)):
        _mu = float(np.mean(_arr))
        _se = float(np.std(_arr, ddof=1) / np.sqrt(len(_arr)))
        _ax_v.errorbar(
            _i, _mu, yerr=_se, fmt="o", color="white",
            markerfacecolor=_col, markeredgecolor="black",
            markersize=8, capsize=4, zorder=5,
        )
        _succ = int(np.sum(_arr > 0))
        _ax_v.text(
            _i, 1.18,
            f"μ={_mu:+.2f}±{_se:.02f}\n{_succ}/{_N_SEEDS} succ",
            ha="center", va="bottom", fontsize=8,
        )
    _ax_v.set_xticks([0, 1, 2])
    _ax_v.set_xticklabels(_names, fontsize=9)
    _ax_v.set_ylabel("return  (−1 hole, +1 goal)")
    _ax_v.set_ylim(-1.3, 1.6)
    _ax_v.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
    _ax_v.axhline(1, color="grey", linewidth=0.5, alpha=0.4)
    _ax_v.axhline(-1, color="grey", linewidth=0.5, alpha=0.4)
    _ax_v.set_title(
        f"Return distribution  (violins, N={_N_SEEDS})", fontsize=10,
    )

    # --- Right: paired-diff strip plot ---------------------------
    _diff_oracle_stale = _oracle_a - _stale_a
    _diff_oracle_replan = _oracle_a - _replan_a
    _diffs = [_diff_oracle_stale, _diff_oracle_replan]
    _diff_names = ["oracle − stale", "oracle − replan"]
    _diff_colors = ["tab:red", "tab:blue"]  # color of the *baseline*
    _rng = np.random.default_rng(0)
    for _i, (_d, _col) in enumerate(zip(_diffs, _diff_colors)):
        _xs = _i + (_rng.uniform(-0.16, 0.16, size=len(_d)))
        _ax_d.scatter(
            _xs, _d, s=18, alpha=0.55, color=_col,
            edgecolor="black", linewidth=0.3,
        )
        _mu = float(np.mean(_d))
        _se = float(np.std(_d, ddof=1) / np.sqrt(len(_d)))
        _wins = int(np.sum(_d > 0))
        _losses = int(np.sum(_d < 0))
        _ax_d.errorbar(
            _i, _mu, yerr=_se, fmt="D", color="white",
            markerfacecolor="black", markeredgecolor="black",
            markersize=7, capsize=5, zorder=5,
        )
        _ax_d.text(
            _i, 2.3,
            f"μ_diff={_mu:+.2f}±{_se:.02f}\n"
            f"oracle wins {_wins}/{_N_SEEDS}  ·  "
            f"loses {_losses}/{_N_SEEDS}",
            ha="center", va="bottom", fontsize=8,
        )
    _ax_d.axhline(0, color="grey", linewidth=0.8)
    _ax_d.set_xticks([0, 1])
    _ax_d.set_xticklabels(_diff_names, fontsize=9)
    _ax_d.set_ylabel("per-seed return diff")
    _ax_d.set_ylim(-2.5, 2.7)
    _ax_d.set_title(
        "Paired difference per seed  (above 0 → oracle wins)",
        fontsize=10,
    )

    _short_sched = fl_sched_pick.value.split(" — ")[0].split("(")[0].strip()
    _short_drift = fl_drift_pick.value.split(" — ")[0].split("(")[0].strip()
    _fig.suptitle(
        f"BRIDGE  •  {_short_sched}  ×  {_short_drift}  •  N={_N_SEEDS}",
        fontsize=10,
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #555; background: #f6f6f6;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Watch the oracle walk Bridge — agent + V(s, t) side by side</div></div>

    ### Same pattern as the FrozenLake animation in Section 3, adapted for Bridge. Bridge's `env.render()` is text-only, so we draw a custom matplotlib grid: ice tiles light blue, holes black, goals gold, agent shown as a magenta marker. Slider scrubs through the rollout. Right panel: the oracle's `V(s, t)` heatmap at the agent's current step — same colormap and overlays as the 3-panel comparison above.
    """)
    return


@app.cell
def _(BRIDGE_MAP, BRIDGE_NCOL, BRIDGE_NROW, FL_ACTION_ARROWS, np):
    from matplotlib.colors import ListedColormap as _BridgeAgentCmap

    # Categorical tile palette: 0=ice (F/S), 1=hole, 2=goal.
    _BRIDGE_TILE_CMAP = _BridgeAgentCmap(["#bce4f0", "#1a1a1a", "#ffd700"])
    _BRIDGE_TILE_CODE = {"F": 0, "S": 0, "H": 1, "G": 2}

    def bridge_draw_agent_panel(ax, agent_state, action_taken=None, title=""):
        """Render Bridge grid + agent marker."""
        tile_grid = np.zeros((BRIDGE_NROW, BRIDGE_NCOL), dtype=int)
        for r in range(BRIDGE_NROW):
            for c in range(BRIDGE_NCOL):
                tile_grid[r, c] = _BRIDGE_TILE_CODE[BRIDGE_MAP[r][c]]
        ax.imshow(tile_grid, cmap=_BRIDGE_TILE_CMAP, vmin=-0.5, vmax=2.5)

        for r in range(BRIDGE_NROW):
            for c in range(BRIDGE_NCOL):
                tile = BRIDGE_MAP[r][c]
                if tile == "H":
                    ax.text(c, r, "H", ha="center", va="center",
                            color="red", fontsize=10, fontweight="bold")
                elif tile == "G":
                    ax.text(c, r, "G", ha="center", va="center",
                            color="black", fontsize=11, fontweight="bold")
                elif tile == "S":
                    ax.text(c, r - 0.32, "S", ha="center", va="center",
                            color="black", fontsize=8, alpha=0.55)

        agent_row, agent_col = divmod(int(agent_state), BRIDGE_NCOL)
        ax.scatter(
            [agent_col], [agent_row],
            s=420, c="magenta", edgecolors="black",
            linewidth=1.5, zorder=5,
        )
        if action_taken is not None:
            ax.text(
                agent_col, agent_row,
                FL_ACTION_ARROWS[int(action_taken)],
                ha="center", va="center", color="white",
                fontsize=12, fontweight="bold", zorder=6,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    return (bridge_draw_agent_panel,)


@app.cell
def _(bridge_make_env, bridge_oracle_policy, type_mismatch_checker):
    def bridge_animate_oracle(seed=0):
        """Roll one episode under the oracle policy, recording state path."""
        env = bridge_make_env()
        obs, _ = env.reset(seed=seed)
        state, _ = type_mismatch_checker(obs)
        T = bridge_oracle_policy.shape[0]
        states = [int(state)]
        actions = []
        ts = [0]
        done = trunc = False
        t = 0
        while not (done or trunc):
            a = int(bridge_oracle_policy[min(t, T - 1), int(state)])
            actions.append(a)
            obs, reward, done, trunc, _ = env.step(a)
            state, reward = type_mismatch_checker(obs, reward)
            states.append(int(state))
            t += 1
            ts.append(t)
        return states, actions, ts

    bridge_anim_states, bridge_anim_actions, bridge_anim_ts = bridge_animate_oracle(seed=0)
    return bridge_anim_actions, bridge_anim_states, bridge_anim_ts


@app.cell
def _(bridge_anim_states, mo):
    bridge_anim_idx = mo.ui.slider(
        start=0, stop=max(len(bridge_anim_states) - 1, 1), step=1, value=0,
        label="Bridge rollout step",
    )
    bridge_anim_idx
    return (bridge_anim_idx,)


@app.cell
def _(
    BRIDGE_NCOL,
    BRIDGE_NROW,
    bridge_anim_actions,
    bridge_anim_idx,
    bridge_anim_states,
    bridge_anim_ts,
    bridge_draw_agent_panel,
    bridge_draw_value_panel,
    bridge_oracle_V,
    bridge_oracle_policy,
    fl_drift_pick,
    fl_sched_pick,
    plt,
):
    _i = min(bridge_anim_idx.value, len(bridge_anim_states) - 1)
    _t = bridge_anim_ts[_i] if _i < len(bridge_anim_ts) else bridge_anim_ts[-1]
    _t_clamped = min(_t, bridge_oracle_policy.shape[0] - 1)
    _vmax = max(float(bridge_oracle_V[0].max()), 1e-3)
    _action_taken = (
        bridge_anim_actions[_i - 1]
        if 0 < _i <= len(bridge_anim_actions) else None
    )
    _action_str = (
        ["←", "↓", "→", "↑"][_action_taken]
        if _action_taken is not None else "—"
    )

    _fig, _axes = plt.subplots(
        1, 2, figsize=(15, 4.6),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    bridge_draw_agent_panel(
        _axes[0],
        agent_state=bridge_anim_states[_i],
        action_taken=_action_taken,
        title=(
            f"step {_i}/{len(bridge_anim_states) - 1}   "
            f"action: {_action_str}   t={_t_clamped}"
        ),
    )
    bridge_draw_value_panel(
        _axes[1],
        bridge_oracle_V[_t_clamped].reshape(BRIDGE_NROW, BRIDGE_NCOL),
        bridge_oracle_policy[_t_clamped].reshape(BRIDGE_NROW, BRIDGE_NCOL),
        f"Oracle V(s, t={_t_clamped})",
        vmax=_vmax,
    )
    _short_sched = fl_sched_pick.value.split(" — ")[0].split("(")[0].strip()
    _short_drift = fl_drift_pick.value.split(" — ")[0].split("(")[0].strip()
    _fig.suptitle(
        f"BRIDGE oracle rollout  •  {_short_sched}  ×  {_short_drift}",
        fontsize=10,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The values trace the optimal path from start (top-left) to goal
    (bottom-right) under the stationary deterministic dynamics. This
    policy is what we deploy on the drifting env below.
    """)
    return


@app.cell
def _(mo):
    fl_break_k = mo.ui.slider(
        start=0.0, stop=0.3, step=0.01, value=0.1,
        label="drift k (breakage demo)",
    )
    fl_break_k
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Your turn — when does the policy break?</div></div>

    ### Implement a *success-rate* helper that runs `n` episodes at a given drift `k` with the stationary VI policy and returns the fraction that reach the goal. The "Reference solution" tab below holds the worked answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _stub = '''def fl_success_rate(k, n=30):
    """Fraction of n episodes that reach the goal at drift k."""
    # TODO: roll out n episodes at drift k with fl_vi_policy
    #       and return the fraction that reach the goal.
    return 0.0
    '''
    _solution = '''def fl_success_rate(k, n=30):
    """Reference: episode-level success fraction."""
    wins = 0
    for s in range(n):
        env = fl_make_env(k)
        ret, _, _ = fl_rollout(env, fl_vi_policy, seed=s)
        wins += int(ret > 0)
    return wins / n
    '''
    fl_success_editor = mo.ui.code_editor(
        value=_stub, language="python", min_height=10,
    )
    _ref = mo.ui.code_editor(
        value=_solution, language="python", disabled=True, min_height=10,
    )
    mo.ui.tabs({
        "✏️ Your code": fl_success_editor,
        "💡 Reference solution": _ref,
    })
    return (fl_success_editor,)


@app.cell
def _(fl_make_env, fl_rollout, fl_success_editor, fl_vi_policy):
    """Compile the user's `fl_success_rate` definition. We inject the
    rollout primitives + the stationary VI policy so the editor's code
    can call them directly without having to re-import. Surface
    ``fl_success_rate`` (the callable or None on compile error) for
    the verifier cell below."""
    _ns = {
        "fl_make_env": fl_make_env,
        "fl_rollout": fl_rollout,
        "fl_vi_policy": fl_vi_policy,
    }
    try:
        exec(fl_success_editor.value, _ns)
        fl_success_rate = _ns.get("fl_success_rate")
        if fl_success_rate is None:
            raise NameError("define a function named fl_success_rate")
        _msg = "✓ fl_success_rate compiled"
    except Exception as _e:
        fl_success_rate = None
        _msg = f"⚠ {type(_e).__name__}: {_e}"
    _msg
    return (fl_success_rate,)


@app.cell
def _(fl_success_rate, mo, np):
    if fl_success_rate is None:
        _out = mo.md("⚠ Fix the function above to see the success-rate table.")
    else:
        _ks_probe = np.linspace(0.0, 0.3, 7)
        _success = [(float(k), fl_success_rate(k, n=30)) for k in _ks_probe]
        _out = mo.md(
            "| k | success rate |\n|---|---|\n"
            + "\n".join(f"| {k:.2f} | {sr:.2%} |" for k, sr in _success)
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    fl_vi_run_sweep = mo.ui.run_button(label="Run stale-VI sweep")
    mo.vstack([
        mo.md(r"""
        <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Sweep — stale VI returns vs drift</div></div>

        ### Same shape as Section 1's random sweep, but using the stationary VI policy. Stashed as `fl_vi_sweep` and saved to `results/nb_2_frozenlake_vi.json`.
        """),
        fl_vi_run_sweep,
    ])
    return (fl_vi_run_sweep,)


@app.cell
def _(
    fl_make_env,
    fl_rollout,
    fl_vi_policy,
    fl_vi_run_sweep,
    mo,
    np,
    save_sweep,
):
    import time as _time
    fl_vi_sweep = None
    fl_vi_sweep_path = None
    if not fl_vi_run_sweep.value:
        _status = mo.md(
            "_Click **Run stale-VI sweep** above (~10 s)._"
        )
    else:
        _t0 = _time.time()
        _ks = np.linspace(0.0, 0.3, 7)
        _returns_by_k = {}
        for _k in mo.status.progress_bar(
            _ks, title="Stale-VI sweep", subtitle="rolling out…",
            remove_on_exit=True,
        ):
            _returns = []
            for _ep in range(30):
                _env = fl_make_env(_k)
                _ret, _, _ = fl_rollout(_env, fl_vi_policy, seed=_ep)
                _returns.append(float(_ret))
            _returns_by_k[float(_k)] = _returns
        fl_vi_sweep = {
            "env": "FrozenLake-v1",
            "param": "P",
            "scheduler": "ContinuousScheduler",
            "update_fn": "DistributionDecrementUpdate",
            "policy": "stationary-vi",
            "n_episodes": 30,
            "returns_by_k": _returns_by_k,
        }
        fl_vi_sweep_path = save_sweep("nb_2_frozenlake_vi.json", fl_vi_sweep)
        _status = mo.md(
            f"✓ Stale-VI sweep finished in **{_time.time() - _t0:.1f} s** — "
            f"7 drift levels × 30 episodes. Saved to `{fl_vi_sweep_path}`."
        )
    _status
    return (fl_vi_sweep,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Same phenomenon, continuous control — CartPole gravity drift</div></div>

    ### Visual generalization. A hand-tuned linear policy that balances CartPole at default gravity. As gravity drifts up, the controller falls outside its gain margin. Slider only — no exercise, no sweep.
    """)
    return


@app.cell
def _(np):
    CP_POLICY_W = np.array([0.1, 1.0, 5.0, 2.0])

    def cp_linear_policy(state):
        score = float(np.dot(CP_POLICY_W, np.asarray(state, dtype=float)))
        return 1 if score > 0 else 0

    return (cp_linear_policy,)


@app.cell
def _(mo):
    cp_demo_k = mo.ui.slider(
        start=0.0, stop=2.0, step=0.05, value=0.5,
        label="CartPole gravity drift k",
    )
    cp_demo_k
    return (cp_demo_k,)


@app.cell
def _(
    ContinuousScheduler,
    IncrementUpdate,
    NSClassicControlWrapper,
    cp_demo_k,
    cp_linear_policy,
    gym,
    np,
    plt,
    type_mismatch_checker,
):
    def _cp_make(k):
        env = gym.make("CartPole-v1")
        return NSClassicControlWrapper(
            env,
            {"gravity": IncrementUpdate(scheduler=ContinuousScheduler(), k=float(k))},
        )

    _MAX_STEPS = 500           # CartPole-v1 max-episode-length
    _GRAVITY_YLIM = (0.0, 60.0)  # default 9.8 + drift; fixed across slider positions

    def _cp_rollout(env, seed):
        obs, _ = env.reset(seed=seed)
        state, _ = type_mismatch_checker(obs)
        done = trunc = False
        gtrace = np.full(_MAX_STEPS, np.nan, dtype=np.float64)
        cum_r = np.full(_MAX_STEPS, np.nan, dtype=np.float64)
        ep_return = 0.0
        for step in range(_MAX_STEPS):
            if done or trunc:
                break
            action = cp_linear_policy(state)
            obs, reward, done, trunc, _ = env.step(action)
            state, reward = type_mismatch_checker(obs, reward)
            gtrace[step] = float(env.unwrapped.gravity)
            ep_return += float(reward)
            cum_r[step] = ep_return
        return ep_return, gtrace, cum_r

    _seeds = list(range(5))
    _results = [_cp_rollout(_cp_make(cp_demo_k.value), _s) for _s in _seeds]

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
    for _, _g, _ in _results:
        _axes[0].plot(_g, alpha=0.6)
    _axes[0].set_xlabel("step")
    _axes[0].set_ylabel("gravity")
    _axes[0].set_xlim(0, _MAX_STEPS)
    _axes[0].set_ylim(*_GRAVITY_YLIM)
    _axes[0].set_title(f"Gravity over time (k={cp_demo_k.value:.2f})")

    for _, _, _cr in _results:
        _axes[1].plot(_cr, alpha=0.6)
    _returns = [r for r, _, _ in _results]
    _axes[1].axhline(np.mean(_returns), color="k", linestyle="--", alpha=0.4,
                     label=f"mean return = {np.mean(_returns):.0f}")
    _axes[1].set_xlabel("step")
    _axes[1].set_ylabel("cumulative reward")
    _axes[1].set_xlim(0, _MAX_STEPS)
    _axes[1].set_ylim(0, _MAX_STEPS)        # axis fixed so curves are comparable across k
    _axes[1].set_title(f"Cumulative reward over time (k={cp_demo_k.value:.2f})")
    _axes[1].legend(fontsize=8, loc="lower right")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #555; background: #f6f6f6;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Animated rollout — watch the pole tip over</div></div>

    ### Same gravity-drift slider above; this cell rolls one episode under the hand-tuned linear policy and lets you scrub frame-by-frame. Drawn in matplotlib (no pygame) so it works under `marimo run`.
    """)
    return


@app.cell
def _(np, plt):
    def cp_draw_state(ax, state, *, gravity=None, action=None, title=""):
        """Draw CartPole at the given state vector."""
        x, _x_dot, theta, _theta_dot = state
        ax.axhline(y=0, color="#222", linewidth=1.2)
        cart_w, cart_h = 0.5, 0.30
        cart = plt.Rectangle(
            (x - cart_w / 2, -cart_h / 2), cart_w, cart_h,
            facecolor="#1a1a1a", edgecolor="black",
            linewidth=1.5, zorder=3,
        )
        ax.add_patch(cart)
        pole_length = 1.0
        tip_x = x + pole_length * np.sin(theta)
        tip_y = (cart_h / 2) + pole_length * np.cos(theta)
        ax.plot([x, tip_x], [cart_h / 2, tip_y],
                color="#dd8452", linewidth=6,
                solid_capstyle="round", zorder=4)
        ax.scatter([tip_x], [tip_y], s=120,
                   facecolors="#dd8452", edgecolors="black",
                   linewidths=1.2, zorder=5)
        if action is not None:
            ax.annotate(
                "→" if action == 1 else "←",
                xy=(x, -cart_h / 2 - 0.12),
                ha="center", va="top",
                fontsize=18, fontweight="bold",
                color="tab:blue" if action == 1 else "tab:red",
            )
        if gravity is not None:
            ax.text(
                0.02, 0.98, f"g = {gravity:.2f}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=10, fontfamily="monospace",
                bbox={"facecolor": "white", "alpha": 0.7,
                      "edgecolor": "none", "pad": 3},
            )
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-0.6, 1.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=10)

    return (cp_draw_state,)


@app.cell
def _(
    ContinuousScheduler,
    IncrementUpdate,
    NSClassicControlWrapper,
    cp_demo_k,
    cp_linear_policy,
    gym,
    type_mismatch_checker,
):
    def cp_animate_rollout(seed=0, max_steps=200):
        # No render_mode — no pygame, no SDL, no Cocoa drama.
        env = gym.make("CartPole-v1")
        ns_env = NSClassicControlWrapper(
            env,
            {"gravity": IncrementUpdate(
                scheduler=ContinuousScheduler(),
                k=float(cp_demo_k.value),
            )},
        )
        obs, _ = ns_env.reset(seed=seed)
        state, _ = type_mismatch_checker(obs)
        states = [tuple(state)]
        actions = []
        gravities = [float(env.unwrapped.gravity)]
        done = trunc = False
        steps = 0
        while not (done or trunc) and steps < max_steps:
            a = cp_linear_policy(state)
            actions.append(int(a))
            obs, _, done, trunc, _ = ns_env.step(a)
            state, _ = type_mismatch_checker(obs, 0.0)
            states.append(tuple(state))
            gravities.append(float(env.unwrapped.gravity))
            steps += 1
        return states, actions, gravities

    cp_anim_states, cp_anim_actions, cp_anim_gravities = cp_animate_rollout(seed=0)
    return cp_anim_actions, cp_anim_gravities, cp_anim_states


@app.cell
def _(cp_anim_states, mo):
    cp_anim_idx = mo.ui.slider(
        start=0, stop=max(len(cp_anim_states) - 1, 1), step=1, value=0,
        label="CartPole rollout step",
    )
    cp_anim_idx
    return (cp_anim_idx,)


@app.cell
def _(
    cp_anim_actions,
    cp_anim_gravities,
    cp_anim_idx,
    cp_anim_states,
    cp_draw_state,
    np,
    plt,
):
    _i = min(cp_anim_idx.value, len(cp_anim_states) - 1)
    _state = cp_anim_states[_i]
    _action = (
        cp_anim_actions[_i - 1]
        if 0 < _i <= len(cp_anim_actions) else None
    )
    _gravity = (
        cp_anim_gravities[_i]
        if _i < len(cp_anim_gravities) else None
    )

    # full state traces over the rollout
    _xs = np.array([s[0] for s in cp_anim_states])
    _x_dots = np.array([s[1] for s in cp_anim_states])
    _thetas = np.array([s[2] for s in cp_anim_states])
    _theta_dots = np.array([s[3] for s in cp_anim_states])
    _failed_at = next(
        (j for j, t in enumerate(_thetas) if abs(t) > 0.2095),
        None,
    )

    _fig = plt.figure(figsize=(10, 6.4))
    _gs = _fig.add_gridspec(
        4, 1, height_ratios=[4, 1, 1, 1], hspace=0.35,
    )
    _ax_cart = _fig.add_subplot(_gs[0])
    _ax_x = _fig.add_subplot(_gs[1])
    _ax_theta = _fig.add_subplot(_gs[2])
    _ax_act = _fig.add_subplot(_gs[3])

    # top: cart drawing
    cp_draw_state(
        _ax_cart, _state,
        gravity=_gravity, action=_action,
        title=(
            f"step {_i}/{len(cp_anim_states) - 1}"
            + (f"   ⚠ pole crossed ±12° at step {_failed_at}"
               if _failed_at is not None else "")
        ),
    )

    # panel 1: cart position + velocity
    _ax_x.plot(_xs, color="tab:blue", linewidth=1.4, label="x")
    _ax_x.plot(_x_dots, color="tab:cyan",
               linewidth=1, alpha=0.6, label="ẋ")
    _ax_x.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    _ax_x.axvline(_i, color="black", linewidth=1)
    _ax_x.set_ylabel("cart\nx, ẋ", fontsize=8)
    _ax_x.set_xticks([])
    _ax_x.legend(fontsize=7, loc="upper right", ncol=2, frameon=False)
    _ax_x.tick_params(labelsize=7)

    # panel 2: pole angle + angular velocity (with failure band)
    _ax_theta.axhspan(0.2095, 0.5, color="tab:red", alpha=0.10)
    _ax_theta.axhspan(-0.5, -0.2095, color="tab:red", alpha=0.10)
    _ax_theta.axhline(0.2095, color="tab:red", linewidth=0.6, alpha=0.4)
    _ax_theta.axhline(-0.2095, color="tab:red", linewidth=0.6, alpha=0.4)
    _ax_theta.plot(_thetas, color="tab:red", linewidth=1.4, label="θ")
    _ax_theta.plot(_theta_dots, color="tab:orange",
                   linewidth=1, alpha=0.6, label="θ̇")
    _ax_theta.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    _ax_theta.axvline(_i, color="black", linewidth=1)
    _ax_theta.set_ylabel("pole\nθ, θ̇", fontsize=8)
    _ax_theta.set_xticks([])
    _ax_theta.legend(fontsize=7, loc="upper right", ncol=2, frameon=False)
    _ax_theta.tick_params(labelsize=7)

    # panel 3: action sequence (binary, 0=left, 1=right)
    if cp_anim_actions:
        _act_arr = np.array(cp_anim_actions)
        _ax_act.scatter(
            np.arange(len(_act_arr)), _act_arr,
            c=["tab:red" if a == 0 else "tab:blue" for a in _act_arr],
            s=6, alpha=0.7,
        )
    _ax_act.axvline(_i, color="black", linewidth=1)
    _ax_act.set_ylim(-0.4, 1.4)
    _ax_act.set_yticks([0, 1])
    _ax_act.set_yticklabels(["←", "→"], fontsize=10)
    _ax_act.set_ylabel("action", fontsize=8)
    _ax_act.set_xlabel("step", fontsize=8)
    _ax_act.tick_params(labelsize=7)

    _fig.subplots_adjust(top=0.94, bottom=0.07, left=0.07, right=0.98)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(np, type_mismatch_checker, value_iteration):
    def fl_replan_rollout(env, initial_policy, seed=0):
        """Notification-driven replanning rollout (myopic-replan agent).

        On each step we:
          1. take the action prescribed by `policy[state]`;
          2. read the change-notification side-channel;
          3. if `env_change["P"]` fires, request a *current-model
             snapshot* via `env.get_planning_env()` (notification
             level 3 — full env model) and run **full stationary VI**
             on it. Replace `policy` with the newly extracted greedy
             policy.

        Step 3 is a fresh infinite-horizon Bellman fixed-point on the
        most recently observed transition table — *not* a one-step
        lookahead. The agent plans the whole horizon assuming the
        current model is permanent; foresight enters only via the
        notification-triggered re-solve.

        Going through `get_planning_env()` rather than reading
        `env.unwrapped.P` directly is the canonical NS-Gym contract:
        the planner only sees what the wrapper's notification policy
        permits. With `delta_change_notification=True` the snapshot
        reflects up-to-date params; with it off, you'd plan against
        the initial model.
        """
        policy = np.asarray(initial_policy).copy()
        obs, _ = env.reset(seed=seed)
        state, _ = type_mismatch_checker(obs)
        done = trunc = False
        ep_return = 0.0
        replans = 0
        while not (done or trunc):
            action = int(policy[state])
            obs, reward, done, trunc, _ = env.step(action)
            if isinstance(obs, dict) and obs.get("env_change", {}).get("P", 0):
                planning_env = env.get_planning_env()
                policy, _ = value_iteration(planning_env)  # full stationary VI
                replans += 1
            state, reward = type_mismatch_checker(obs, reward)
            ep_return += reward
        return ep_return, replans

    return (fl_replan_rollout,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Why stationary algorithms break — at the action level</div></div>

    ### A success-rate plot tells you *how often* the policy fails. The Q-gap plots tell you *how much value it leaves on the table*. To see *why* it fails — and what's so special about non-stationary problems — look at where the policy and the time-indexed oracle **disagree on the action** each step.

    ### **Left: action-disagreement over time.** At step $t$, what fraction of seeds chose an action the oracle would *not* have chosen at the agent's current state? Stale-VI is solving against $P_0$ forever, so as the env drifts the disagreement **grows monotonically**. Replan-VI re-solves on every notification and stays low. Oracle is zero by construction (it *is* the reference). The slope of the stale curve is the visual signature of "the policy went stale."

    ### **Right: empirical CDF of cumulative returns.** $\hat F(x) = \mathbb P[\,\text{return} \le x\,]$ over $N$ seeds. A sparse-reward FrozenLake return is 0 (failed) or 1 (reached goal), so each curve is a step function with the height at $x = 0$ equal to the **failure rate**. **What the eCDF tells you that a mean doesn't:**

    - **Distribution shape.** Two methods can have the same mean
      ### $\bar R$ via very different shapes: one wins big sometimes and dies often (bimodal), another scores small reliably. A bar chart hides this; the eCDF makes the failure mode literal.
    - **Risk profile.** A method whose curve rises steeply on the
      ### left (lots of mass near 0) is *unreliable* in a way that averaging suppresses.
    - **Stochastic dominance.** If method A's eCDF lies entirely
      ### to the right of method B's, A *first-order stochastically dominates* B — for any threshold $x$, A is at least as likely to clear it as B.

    > This is the gap that NS-MDP-specific algorithms close: re-planning,
    > worst-case planning, or context-conditioned policies all aim to
    > keep the disagreement curve low so the eCDF of returns sits as
    > far right as possible — even when the env drifts.
    """)
    return


@app.cell
def _(
    fl_adapt_k,
    fl_make_env,
    fl_oracle_horizon,
    fl_oracle_policy,
    fl_vi_policy,
    np,
    plt,
    type_mismatch_checker,
    value_iteration,
):
    _N_SEEDS = 50
    _T_MAX = int(fl_oracle_horizon)
    _k = float(fl_adapt_k.value)

    def _oracle_action(t, state):
        _T = fl_oracle_policy.shape[0]
        return int(fl_oracle_policy[min(t, _T - 1), int(state)])

    # fl_vi_policy is a callable (state -> action) elsewhere in this notebook;
    # capture its t=0 actions as a 16-vector once so we can subscript it for
    # both stale and the replan-warm-start.
    _stale_pi_arr = np.asarray(
        [int(fl_vi_policy(_s)) for _s in range(16)],
        dtype=np.int64,
    )

    def _stale_factory():
        # No state, no post-step hook
        return (lambda s, t, env: int(_stale_pi_arr[int(s)]), None)

    def _oracle_factory():
        return (lambda s, t, env: _oracle_action(t, int(s)), None)

    def _replan_factory():
        # Mutable policy; refit when env_change notification fires.
        _state = {"policy": _stale_pi_arr.copy()}
        def _picker(s, t, env):
            return int(_state["policy"][int(s)])
        def _after_step(obs, env):
            if isinstance(obs, dict) and obs.get("env_change", {}).get("P", 0):
                _new_policy, _ = value_iteration(env.get_planning_env())
                _state["policy"] = np.asarray(_new_policy).reshape(-1).astype(np.int64)
        return (_picker, _after_step)

    def _rollout(picker_factory, n_seeds):
        _sum_disagree = np.zeros(_T_MAX, dtype=np.float64)
        _count = np.zeros(_T_MAX, dtype=np.float64)
        _final_r = np.zeros(n_seeds, dtype=np.float64)
        for _s in range(n_seeds):
            _picker, _after_step = picker_factory()
            _env = fl_make_env(_k)
            _obs, _ = _env.reset(seed=_s)
            _state, _ = type_mismatch_checker(_obs)
            _done = _trunc = False
            _t = 0
            while not (_done or _trunc) and _t < _T_MAX:
                _a_taken = int(_picker(int(_state), _t, _env))
                _a_oracle = _oracle_action(_t, int(_state))
                _sum_disagree[_t] += 1.0 if (_a_taken != _a_oracle) else 0.0
                _count[_t] += 1.0
                _obs, _reward, _done, _trunc, _ = _env.step(_a_taken)
                if _after_step is not None:
                    _after_step(_obs, _env)
                _state, _reward = type_mismatch_checker(_obs, _reward)
                _final_r[_s] += float(_reward)
                _t += 1
        with np.errstate(invalid="ignore"):
            _disagreement = _sum_disagree / np.maximum(_count, 1)
        return _disagreement, _count, _final_r

    _planners = [
        ("stale-VI",  "tab:red",   _stale_factory),
        ("replan-VI", "tab:blue",  _replan_factory),
        ("oracle-VI", "tab:green", _oracle_factory),
    ]

    _data = {}
    for _name, _color, _fac in _planners:
        _disagree, _count, _returns = _rollout(_fac, _N_SEEDS)
        _data[_name] = {
            "disagree": _disagree, "count": _count,
            "returns": _returns, "color": _color,
        }

    _fig, (_ax_d, _ax_c) = plt.subplots(
        1, 2, figsize=(12.5, 4.4), layout="constrained",
    )

    # ----- LEFT: disagreement-over-time -----
    for _name, _color, _ in _planners:
        _info = _data[_name]
        _mask = _info["count"] > 0
        _xs = np.arange(_T_MAX)[_mask]
        _ax_d.plot(_xs, _info["disagree"][_mask], label=_name, color=_color,
                   linewidth=1.8, marker="o", markersize=3)
    _ax_d.set_xlabel("step")
    _ax_d.set_ylabel(r"$\Pr(a_\mathrm{taken} \neq a^*_t)$  given the agent reached step $t$")
    _ax_d.set_ylim(-0.02, 1.02)
    _ax_d.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
    _ax_d.set_title(
        f"Action disagreement vs time-indexed oracle  "
        f"(k={_k:.2f}, N={_N_SEEDS} seeds)",
        fontsize=10,
    )
    _ax_d.legend(fontsize=8, loc="lower right")
    _ax_d.grid(alpha=0.25)

    # ----- RIGHT: empirical CDF -----
    for _name, _color, _ in _planners:
        _info = _data[_name]
        _rs = np.sort(_info["returns"])
        _ys = np.arange(1, len(_rs) + 1) / len(_rs)
        # Pad with a (min - eps, 0) point so the step starts cleanly at 0.
        _xs_aug = np.concatenate([[float(_rs.min()) - 0.05], _rs])
        _ys_aug = np.concatenate([[0.0], _ys])
        _ax_c.step(_xs_aug, _ys_aug, where="post", label=_name, color=_color,
                   linewidth=1.8)
    _ax_c.set_xlabel("cumulative return per episode")
    _ax_c.set_ylabel(r"$\hat F(x) = \Pr(\mathrm{return} \leq x)$")
    _ax_c.set_ylim(-0.02, 1.02)
    _ax_c.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
    _ax_c.axhline(1, color="grey", linewidth=0.5, alpha=0.4)
    _ax_c.set_title(
        f"Empirical CDF of returns  (k={_k:.2f}, N={_N_SEEDS} seeds)",
        fontsize=10,
    )
    _ax_c.legend(fontsize=8, loc="lower right")
    _ax_c.grid(alpha=0.25)

    _fig.suptitle(
        "Why stationary algorithms break — action-level + outcome-shape evidence",
        fontsize=11, fontweight=600,
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <div style="
        margin: 5em 0 1.5em 0;
        padding: 2.5em 2em;
        border-top: 8px solid #1a1a1a;
        border-bottom: 8px solid #1a1a1a;
        background: linear-gradient(180deg, #f5f5f5, #fafafa);
    ">
      <div style="font-size: 0.9em; opacity: 0.55;
                  letter-spacing: 0.2em; font-weight: 600;
                  color: #1a1a1a;">
        MODULE 3 &middot; 11:45 · 12:00
      </div>
      <div style="font-size: 2.7em; font-weight: 800;
                  margin-top: 0.25em; line-height: 1.05;">
        Evaluating Algorithms on NS-MDPs
      </div>
      <div style="font-size: 1.05em; opacity: 0.72;
                  margin-top: 0.7em; max-width: 56em;">
        Read sweep JSONs and compare four policies side-by-side: random, stale-VI, replan-VI, and oracle-VI. What does each tell us about a policy's robustness to drift?
      </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Cross-policy evaluation on FrozenLake

    Four policies on the same NS-FrozenLake:

    - **Random** — Section 1's baseline (`fl_sweep`).
    - **Stale VI** — solved on the stationary env, deployed unchanged
      on the drifting one (`fl_vi_sweep`).
    - **Re-plan VI** — Section 3's notification-driven re-planner
      (`fl_replan_sweep["returns_by_k_adaptive"]`).
    - **Oracle VI** — Section 3's time-augmented JIT planner with
      perfect foresight on the drift schedule (run inline below).

    Run the sweep buttons above; this section populates live as each
    one finishes.
    """)
    return


@app.cell
def _(fl_replan_sweep, fl_sweep, fl_vi_sweep):
    eval_runs = {}
    if fl_sweep is not None:
        eval_runs["random"] = fl_sweep["returns_by_k"]
    if fl_vi_sweep is not None:
        eval_runs["stale-VI"] = fl_vi_sweep["returns_by_k"]
    if fl_replan_sweep is not None:
        eval_runs["replan-VI"] = fl_replan_sweep["returns_by_k_adaptive"]
    eval_missing = [
        label for label, dataset in [
            ("random (Section 1)", fl_sweep),
            ("stale-VI (Section 2)", fl_vi_sweep),
            ("replan-VI (Section 3)", fl_replan_sweep),
            ("oracle-VI (inline)", None),
        ] if dataset is None
    ]
    return eval_missing, eval_runs


@app.cell(hide_code=True)
def _(eval_missing, mo):
    fl_eval_resweep = mo.ui.run_button(
        label=(
            f"▶ Run missing sweeps inline ({len(eval_missing)})"
            if eval_missing else "▶ Re-run all four sweeps inline"
        ),
    )
    if eval_missing:
        _msg = mo.md(
            "**Missing sweeps:** "
            + ", ".join(eval_missing)
            + ".  No need to scroll back — click below to run them inline."
        )
    else:
        _msg = mo.md("All four sweeps loaded — see plot below.")
    mo.vstack([_msg, fl_eval_resweep])
    return (fl_eval_resweep,)


@app.cell
def _(
    build_ns_tensors,
    eval_missing,
    fl_eval_resweep,
    fl_make_env,
    fl_oracle_gamma,
    fl_oracle_horizon,
    fl_oracle_rollout,
    fl_random_policy,
    fl_replan_rollout,
    fl_rollout,
    fl_stationary_policy,
    fl_vi_policy,
    mo,
    np,
    ns_oracle_vi,
):
    import time as _time
    eval_runs_inline = {}
    if not fl_eval_resweep.value:
        _eval_status = mo.md(
            "_Click the button above to run any missing sweeps inline "
            "(~10–30 s depending on what's missing)._"
        )
    else:
        _t0 = _time.time()
        _missing_label_to_key = {
            "random (Section 1)": "random",
            "stale-VI (Section 2)": "stale-VI",
            "replan-VI (Section 3)": "replan-VI",
            "oracle-VI (inline)": "oracle-VI",
        }
        # If everything is already present, the button means "re-run all".
        if eval_missing:
            _todo = [
                _missing_label_to_key[m] for m in eval_missing
                if m in _missing_label_to_key
            ]
        else:
            _todo = list(_missing_label_to_key.values())

        _ks = np.linspace(0.0, 0.3, 7)
        _N_EP = 30
        for _key in _todo:
            _runs_by_k = {}
            for _k in mo.status.progress_bar(
                _ks, title=f"Inline sweep — {_key}",
                subtitle="rolling out…", remove_on_exit=True,
            ):
                _returns = []
                if _key == "random":
                    for _ep in range(_N_EP):
                        _env = fl_make_env(_k)
                        _ret, _, _ = fl_rollout(_env, fl_random_policy, seed=_ep)
                        _returns.append(float(_ret))
                elif _key == "stale-VI":
                    for _ep in range(_N_EP):
                        _env = fl_make_env(_k)
                        _ret, _, _ = fl_rollout(_env, fl_vi_policy, seed=_ep)
                        _returns.append(float(_ret))
                elif _key == "replan-VI":
                    for _ep in range(_N_EP):
                        _env = fl_make_env(_k)
                        _ret, _ = fl_replan_rollout(
                            _env, fl_stationary_policy, seed=_ep,
                        )
                        _returns.append(float(_ret))
                elif _key == "oracle-VI":
                    _P, _R = build_ns_tensors(
                        lambda kk=_k: fl_make_env(kk),
                        fl_oracle_horizon, seed=0,
                    )
                    _, _policy = ns_oracle_vi(_P, _R, fl_oracle_gamma)
                    for _ep in range(_N_EP):
                        _env = fl_make_env(_k)
                        _ret = fl_oracle_rollout(_env, _policy, seed=_ep)
                        _returns.append(float(_ret))
                _runs_by_k[float(_k)] = _returns
            eval_runs_inline[_key] = _runs_by_k
        _eval_status = mo.md(
            f"✓ Inline sweeps finished in **{_time.time() - _t0:.1f} s** — "
            f"ran {', '.join(_todo)} ({_N_EP} eps × 7 drift levels each)."
        )
    _eval_status
    return (eval_runs_inline,)


@app.cell
def _(eval_runs, eval_runs_inline, np, plt):
    # Upstream-computed sweeps win; inline results fill gaps.
    _runs = dict(eval_runs_inline)
    _runs.update(eval_runs)
    if _runs:
        _fig, _ax = plt.subplots(figsize=(7, 3.6))
        _colors = {
            "random": "tab:gray",
            "stale-VI": "tab:red",
            "replan-VI": "tab:blue",
            "oracle-VI": "tab:green",
        }
        _markers = {
            "random": "o", "stale-VI": "s", "replan-VI": "^", "oracle-VI": "D",
        }
        for _label, _rbk in _runs.items():
            _ks = sorted(_rbk)
            _means = [np.mean(_rbk[k]) for k in _ks]
            _stds = [np.std(_rbk[k]) for k in _ks]
            _ax.errorbar(_ks, _means, yerr=_stds,
                         marker=_markers.get(_label, "o"),
                         color=_colors.get(_label),
                         capsize=3, label=_label)
        _ax.set_xlabel("drift k")
        _ax.set_ylabel("mean return (1=success)")
        _ax.set_title("FrozenLake — random vs stale-VI vs replan-VI vs oracle-VI")
        _ax.legend()
        _ax.set_ylim(-0.05, 1.05)
        _fig.tight_layout()
        _out = _fig
    else:
        _out = "Run the sweeps above to populate this plot."
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #5a3a8a; background: #f3eef7;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">What to look for</div></div>

    ### - **Random** sets the floor: no model, no plan, no memory. Its return is roughly invariant to drift.
    ### - **Stale-VI** starts strong (perfect on the deterministic env) but degrades sharply as `k` grows.
    ### - **Replan-VI** uses the change notification to re-solve VI on the live `P` table. The gap between stale and replan is the *notification dividend* — same env, same drift, only difference is whether the agent listens to the side-channel.
    ### - **Oracle-VI** is the ceiling: time-augmented finite-horizon VI with perfect foresight on the drift schedule. The gap between replan and oracle is what reactive agents lose by not knowing the future.

    ### The CartPole demo in Module 2 shows the same pattern in continuous control. Same theory, smaller laptop bill on FrozenLake.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <div style="
        margin: 5em 0 1.5em 0;
        padding: 2.5em 2em;
        border-top: 8px solid #1a1a1a;
        border-bottom: 8px solid #1a1a1a;
        background: linear-gradient(180deg, #f5f5f5, #fafafa);
    ">
      <div style="font-size: 0.9em; opacity: 0.55;
                  letter-spacing: 0.2em; font-weight: 600;
                  color: #1a1a1a;">
        MODULE 4 &middot; 12:00 · 12:20
      </div>
      <div style="font-size: 2.7em; font-weight: 800;
                  margin-top: 0.25em; line-height: 1.05;">
        Exploring Adaptive Decision Making
      </div>
      <div style="font-size: 1.05em; opacity: 0.72;
                  margin-top: 0.7em; max-width: 56em;">
        A short survey of representative ways to handle non-stationarity:
        a pure reactive RL family (DQN at three training regimes) and
        three tree-search methods (MCTS, PA-MCTS, RATS) — each
        targeting a different assumption about what the agent knows.
    </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Aproaches to Decision Making in NS-MDPs</div></div>


    ## - There is no single "right" way to do decision making in an NS-MDP — each method bakes in a different assumption about what the agent knows and when.
    ## - The rest of this module walks through how you can use NS-Gym to develop and test appraoches under varying conditions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">A survey of approaches we'll compare</div></div>

    ### Five planners, each baking a different assumption about what the agent knows and when:

    | Family | Method | What it assumes | Decision time |
    |---|---|---|---|
    | **Pure reactive RL** | DDQN — *low slip* ($p{=}0.333$) | one stationary env, train once | µs (forward pass) |
    | | DDQN — *high slip* ($p{=}0.700$) | one stationary env, train once | µs |
    | | DDQN — *contextual* ($p \in [0.5, 1.0]$) | drift, but $\theta$ is observable at train and test | µs |
    | **Online tree search** | **MCTS** | a generative model; balance exploration/exploitation per call | budget × planning steps |
    | | **PA-MCTS** | (a) trained policy on the base env, (b) $\theta$ shifts once and is *notified*, but **no time to retrain** — fuse the prior with online search | budget × planning steps |
    | | **RATS** | drift is bounded but **adversarial**; want a policy that's robust to *any* nearby model | LP per chance node |

    ### Read the tree-search row as a progression: nothing assumed (MCTS) → drift is announced once (PA-MCTS) → drift is bounded but adversarial (RATS). The reactive RL family is fast but assumes stationarity at train time. Detailed elevator pitches below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">MCTS — Monte Carlo Tree Search (Kocsis & Szepesvári, 2006)</div></div>

    ### Builds a search tree rooted at the current state by repeating four phases until the simulation budget runs out — *select* a path down the tree using **UCB1** ($a^* = \arg\max_a \bar{Q}(s,a) + c \sqrt{\ln N(s)/N(s,a)}$), *expand* the leaf with one new action, *simulate* a random rollout to terminal/horizon, and *back-up* the discounted return along the path. The action played at the root is the most-visited child.

    ### MCTS is **anytime** (more budget → better action), works **without a value function**, and only needs a *generative model* of the env (`env.step` and `env.reset`). NS-Gym's `MCTS` class re-builds the tree each step against `env.get_planning_env()`, so it picks up parameter drift for free as long as the planning env is fresh.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.image("assets/MCTS-steps.svg", width=620),
        mo.md(
            "<div style='text-align:center; font-size:0.78em; opacity:0.6;'>"
            "MCTS phases: <i>Selection · Expansion · Simulation · Backpropagation</i>. "
            "Image by Robert Moss — Own work, CC BY-SA 4.0, "
            "<a href='https://commons.wikimedia.org/w/index.php?curid=88889583'>"
            "commons.wikimedia.org/w/index.php?curid=88889583</a>."
            "</div>"
        ),
    ], align="center", gap=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Try it — same MCTS, three notification levels</div></div>

    ### Same env, same drift, same MCTS — only the **notification level** changes. The agent's only difference is *what `get_planning_env()` hands it*: the **initial** snapshot (no delta info, levels 1–2) or the **live** snapshot at the current `t` (delta info, level 3).

    ### We average over 8 seeds with a moderate MCTS budget (`d=20, m=80`); expect ~30 s of compute.

    ### The MCTS interaction loop

    ### The planner is just MCTS from `ns_gym.benchmark_algorithms`, rebuilt every step from `env.get_planning_env()`:

    ```python
    from ns_gym.benchmark_algorithms import MCTS

    obs, _ = env.reset(seed=0)
    state = obs["state"]                       # NS-Gym dict obs
    done = trunc = False
    while not (done or trunc):
        # The planning env's P table reflects the notification level:
        #   none / change_notification -> initial-snapshot P
        #   delta_change_notification  -> current-snapshot P
        penv = env.get_planning_env()
        agent = MCTS(penv, state=state, d=20, m=80, c=1.4, gamma=0.95)
        action, _ = agent.search()
        obs, reward, done, trunc, _ = env.step(int(action))
        state = obs["state"]
    ```

    > **What you should look at first:** the **right panel** —
    > what the planner *thinks* the slip distribution is. With
    > `none` / `change_notification`, the planner is stuck on
    > $\theta_0$ forever; with `delta_change_notification` it
    > tracks the live $\theta_t$ (drift visible from step 1).
    > The right panel is the educational core.
    >
    > **The left panel is more subtle.** On 4×4 FrozenLake the
    > optimal path is *forced* (any path crowds holes), so
    > planning against the live slippery model doesn't unlock
    > a safer route — it just adds rollout variance to MCTS
    > Q-estimates, which can hurt small-budget search. The
    > deterministic-snapshot planners (`none` / `change`)
    > sometimes ride the optimal-on-deterministic plan to a
    > goal that the slip-aware planner second-guesses itself
    > out of. **Notification level changes what the planner
    > perceives; whether that perception helps depends on the
    > env.** On a denser-reward / less-pinned env you'd
    > expect the ranking to flip.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    fl_notif_run = mo.ui.run_button(
        label="▶ Roll MCTS at 3 notification levels (8 seeds × 3 levels)"
    )
    fl_notif_run
    return (fl_notif_run,)


@app.cell
def _(
    ContinuousScheduler,
    DistributionDecrementUpdate,
    NSFrozenLakeWrapper,
    fl_init_intended,
    fl_notif_run,
    gym,
    mo,
    np,
    plt,
    type_mismatch_checker,
):
    import time as _time
    if not fl_notif_run.value:
        _out = mo.md(
            "_Click the run button to launch the comparison "
            "(~30 s)._"
        )
    else:
        _t0 = _time.time()
        from ns_gym.benchmark_algorithms import MCTS as _MCTS

        def _make_env(change_notif, delta_notif):
            base = gym.make(
                "FrozenLake-v1", is_slippery=False,
                max_episode_steps=50,
            )
            sched = ContinuousScheduler()
            update_fn = DistributionDecrementUpdate(
                scheduler=sched, k=0.06,
            )
            return NSFrozenLakeWrapper(
                base, {"P": update_fn},
                change_notification=change_notif,
                delta_change_notification=delta_notif,
                initial_prob_dist=init_dist(fl_init_intended.value),
            )

        _levels = [
            ("none",                       False, False, "tab:red"),
            ("change_notification",        True,  False, "tab:orange"),
            ("delta_change_notification",  True,  True,  "tab:green"),
        ]
        _MAX = 30
        _N_SEEDS = 8
        _MCTS_KW = dict(d=20, m=80, c=1.4, gamma=0.95)
        _by_level = {}
        # One progress tick per (level, seed) episode = 24 total.
        _episodes = [
            (_lbl, _cn, _dn, _col, _seed)
            for (_lbl, _cn, _dn, _col) in _levels
            for _seed in range(_N_SEEDS)
        ]
        for _lbl, _cn, _dn, _col, _seed in mo.status.progress_bar(
            _episodes,
            title="MCTS — three notification levels",
            subtitle="rolling out one episode per (level, seed)…",
            remove_on_exit=True,
        ):
            np.random.seed(_seed)
            _env = _make_env(_cn, _dn)
            _obs, _ = _env.reset(seed=_seed)
            _state, _ = type_mismatch_checker(_obs)
            _cur = 0.0
            _done = _trunc = False
            _t = 0
            _per = []
            while not (_done or _trunc) and _t < _MAX:
                _penv = _env.get_planning_env()
                _per.append(float(_penv.transition_prob[0]))
                _agent = _MCTS(
                    _penv, state=int(_state), **_MCTS_KW,
                )
                _a, _ = _agent.search()
                _obs, _r, _done, _trunc, _ = _env.step(int(_a))
                _state, _r = type_mismatch_checker(_obs, _r)
                _cur += float(_r)
                _t += 1
            _bucket = _by_level.setdefault(
                _lbl, {"returns": [], "perceived": None, "color": _col},
            )
            _bucket["returns"].append(_cur)
            if _bucket["perceived"] is None:
                while len(_per) < _MAX:
                    _per.append(_per[-1] if _per else 1.0)
                _bucket["perceived"] = _per

        _fig, (_ax_r, _ax_p) = plt.subplots(
            1, 2, figsize=(11.5, 3.6), layout="constrained",
        )
        # Left: bar chart of mean return ± std across seeds
        _names = list(_by_level.keys())
        _means = [np.mean(_by_level[k]["returns"]) for k in _names]
        _stds = [np.std(_by_level[k]["returns"]) for k in _names]
        _succs = [
            sum(1 for r in _by_level[k]["returns"] if r > 0)
            for k in _names
        ]
        _colors = [_by_level[k]["color"] for k in _names]
        _xs = np.arange(len(_names))
        _ax_r.bar(_xs, _means, yerr=_stds, color=_colors, capsize=4,
                  edgecolor="black", linewidth=0.5)
        for _i, (_m, _succ) in enumerate(zip(_means, _succs)):
            _ax_r.text(
                _xs[_i], _m + 0.04,
                f"μ={_m:.2f}\n{_succ}/{_N_SEEDS} succ",
                ha="center", va="bottom", fontsize=8,
            )
        _ax_r.set_xticks(_xs)
        _ax_r.set_xticklabels(
            ["none", "change", "delta"], fontsize=9,
        )
        _ax_r.set_ylabel("mean episode return")
        _ax_r.set_ylim(0, 1.2)
        _ax_r.set_title(
            f"MCTS return  (mean ± std over {_N_SEEDS} seeds)",
            fontsize=10,
        )
        # Right: planner-perception trace from seed 0 (representative)
        for _k, _info in _by_level.items():
            _ax_p.plot(
                _info["perceived"], label=_k, color=_info["color"],
                linewidth=1.8, marker="o", markersize=3,
            )
        _ax_p.set_xlabel("step")
        _ax_p.set_ylabel(r"$P[\text{intended}]$ in the planning env")
        _ax_p.set_title(
            "What slip prob does the planner think it's facing?  "
            "(seed 0)",
            fontsize=10,
        )
        _ax_p.set_ylim(-0.05, 1.05)
        _ax_p.legend(fontsize=7, loc="lower left")
        _elapsed = _time.time() - _t0
        _summary = "  ·  ".join(
            f"{k.split('_')[0]}: μ={np.mean(_by_level[k]['returns']):.2f}"
            for k in _by_level
        )
        _out = mo.vstack([
            _fig,
            mo.md(
                f"✓ Finished in **{_elapsed:.1f} s** "
                f"({len(_episodes)} episodes total).  {_summary}"
            ),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">PA-MCTS — Policy-Augmented MCTS (Pettet et al., 2024)</div></div>

    ### Mixes a tree-search value estimate with a **pre-trained policy's bootstrapped value** at the root:

    $$
    Q_\text{PA}(s, a) = \alpha \, Q_\text{MCTS}(s, a) + (1 - \alpha)\, Q_\pi(s, a)
    $$

    ### where $Q_\pi$ comes from a deep RL agent trained on the *stationary* base env. The mixing weight $\alpha$ trades off planning depth against policy bias: $\alpha = 1$ is plain MCTS, $\alpha = 0$ is the raw policy. **The setting PA-MCTS was designed for:** you have a policy trained offline on the base env; at deployment a *single big change* in $\theta$ is announced via `change_notification`; you have **no time to retrain**, but do have time to run a short tree search. PA-MCTS lets the prior carry the load where the search hasn't visited, and the search corrects the prior where the new $\theta$ makes it sub-optimal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">RATS — Risk-Averse Tree Search (Lecarpentier & Rachelson, 2019)</div></div>

    ### A **min-max tree search** for NS-MDPs with bounded but *adversarial* parameter drift. Where MCTS samples a single rollout per leaf, RATS treats nature as a worst-case adversary: at chance nodes it solves a small LP that picks the transition kernel $P_t' \in \mathcal{P}$ minimizing the agent's value, subject to a Lipschitz/Wasserstein bound on how far $P_t'$ can move from the current snapshot.

    ### The output is a policy that is **robust** to *any* drift inside the uncertainty ball. The price: RATS is slower than MCTS (LP per chance node), and the value is *pessimistic* — it leaves return on the table when the actual drift is benign.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">DQN / Double DQN (Mnih et al., 2013; van Hasselt et al., 2016)</div></div>

    ### A neural-network policy: $Q_\theta(s, a)$ trained by minimizing the TD error against a slowly-tracked target net. Double DQN decouples action selection and evaluation in the bootstrap target to fix Q-learning's overestimation bias. Once trained, deployment is trivial — `argmax_a Q_\theta(s, a)` per step, no planning, no rollouts.

    ### NS-Gym ships three pre-trained DDQN weight files for FrozenLake, each illustrating a *different policy of the trainer* w.r.t. drift. The plots below show training-time return curves for the three cases:

    1. **Stationary / low slip** ($p_\text{intended} = 0.333$) —
       ### trained on a *highly* slippery env. Learns a defensive policy that hedges; transfers poorly when slip *decreases*.
    2. **Stationary / high intended** ($p_\text{intended} = 0.700$) —
       ### closer to deterministic; learns an aggressive policy that breaks under heavy slip.
    3. **Context-aware DDQN over $p \in [0.50, 1.00]$** — trained
       ### across a *range* of slip values with the slip parameter concatenated into the state. The policy is a *function of* the dynamics — closest to a domain-randomization-style solution to non-stationarity.

    ### The pretrained weights live under `model_weights/`; the tutorial loads them directly with `stable_baselines3.DQN.load`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def _ddqn_card(path, label):
        return mo.vstack(
            [mo.image(f"assets/{path}", width=320),
             mo.md(f"<div style='text-align:center; font-size:0.85em; opacity:0.7;'>{label}</div>")],
            align="center",
        )

    mo.vstack([
        mo.md("<div style='text-align:center; font-size:0.9em; opacity:0.6; letter-spacing:0.15em; font-weight:600;'>DDQN TRAINING — THREE CASES</div>"),
        mo.hstack([
            _ddqn_card("ddqn_stationary_p0.333.png",
                       "Stationary, p=0.333 (high slip)"),
            _ddqn_card("ddqn_stationary_p0.700.png",
                       "Stationary, p=0.700 (low slip)"),
            _ddqn_card("ddqn_contextual_p0.50-1.00.png",
                       "Context-aware, p ∈ [0.50, 1.00]"),
        ], justify="center", gap=1, wrap=True),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Mental model.** MCTS plans without learning. PA-MCTS bolts a
    > learned policy onto the tree as a prior. RATS hardens the tree
    > against adversarial drift. DQN replaces the tree entirely with
    > a learned value net. The kwargs editor below lets you swap among
    > them with one knob (`PAMCTS_ALPHA`, MCTS budget, RATS depth, …).
    """)
    return


@app.cell
def _(mo):
    fl_adapt_k = mo.ui.slider(
        start=0.0, stop=0.3, step=0.01, value=0.15,
        label="drift k (adaptive demo)",
    )
    fl_adapt_k
    return (fl_adapt_k,)


@app.cell
def _(
    fl_adapt_k,
    fl_make_env,
    fl_replan_rollout,
    fl_rollout,
    fl_stationary_policy,
    fl_vi_policy,
    np,
    plt,
):
    _seeds = list(range(30))
    _stale = []
    _replan = []
    _replan_counts = []
    for _s in _seeds:
        _env_a = fl_make_env(fl_adapt_k.value)
        _env_b = fl_make_env(fl_adapt_k.value)
        np.random.seed(_s)
        _ret_a, _, _ = fl_rollout(_env_a, fl_vi_policy, seed=_s)
        np.random.seed(_s)
        _ret_b, _rc = fl_replan_rollout(_env_b, fl_stationary_policy, seed=_s)
        _stale.append(_ret_a)
        _replan.append(_ret_b)
        _replan_counts.append(_rc)

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.2))
    _w = 0.4
    _idx = np.arange(len(_seeds))
    _axes[0].bar(_idx - _w / 2, _stale, _w, color="tab:red",
                 label=f"stale (μ={np.mean(_stale):.2f})")
    _axes[0].bar(_idx + _w / 2, _replan, _w, color="tab:blue",
                 label=f"replan (μ={np.mean(_replan):.2f})")
    _axes[0].set_xlabel("episode")
    _axes[0].set_ylabel("return")
    _axes[0].set_title(f"Stale vs re-planning VI (k={fl_adapt_k.value:.2f})")
    _axes[0].legend(fontsize=8)

    _max_rc = max(_replan_counts) if _replan_counts else 1
    _axes[1].hist(_replan_counts, bins=range(0, _max_rc + 2))
    _axes[1].set_xlabel("re-plans per episode")
    _axes[1].set_ylabel("frequency")
    _axes[1].set_title("Re-plan count distribution")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    fl_replan_run_sweep = mo.ui.run_button(label="Run stale vs re-plan sweep")
    mo.vstack([
        mo.md(r"""
        <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Sweep — stale vs re-planning VI</div></div>

        ### Stashes both curves as `fl_replan_sweep` and saves to `results/nb_3_frozenlake_replan.json`.
        """),
        fl_replan_run_sweep,
    ])
    return (fl_replan_run_sweep,)


@app.cell
def _(
    fl_make_env,
    fl_replan_rollout,
    fl_replan_run_sweep,
    fl_rollout,
    fl_stationary_policy,
    fl_vi_policy,
    mo,
    np,
    save_sweep,
):
    import time as _time
    fl_replan_sweep = None
    fl_replan_sweep_path = None
    if not fl_replan_run_sweep.value:
        _status = mo.md(
            "_Click **Run stale vs re-plan sweep** above (~30 s)._"
        )
    else:
        _t0 = _time.time()
        _ks = np.linspace(0.0, 0.3, 7)
        _stale_by_k = {}
        _replan_by_k = {}
        for _k in mo.status.progress_bar(
            _ks, title="Stale vs replan sweep",
            subtitle="rolling out both planners…",
            remove_on_exit=True,
        ):
            _stale = []
            _replan = []
            for _ep in range(30):
                _env_a = fl_make_env(_k)
                _env_b = fl_make_env(_k)
                _ra, _, _ = fl_rollout(_env_a, fl_vi_policy, seed=_ep)
                _rb, _ = fl_replan_rollout(_env_b, fl_stationary_policy, seed=_ep)
                _stale.append(float(_ra))
                _replan.append(float(_rb))
            _stale_by_k[float(_k)] = _stale
            _replan_by_k[float(_k)] = _replan
        fl_replan_sweep = {
            "env": "FrozenLake-v1",
            "param": "P",
            "scheduler": "ContinuousScheduler",
            "update_fn": "DistributionDecrementUpdate",
            "policy": "stationary-vi",
            "adaptive_policy": "replan-vi",
            "n_episodes": 30,
            "returns_by_k": _stale_by_k,
            "returns_by_k_adaptive": _replan_by_k,
        }
        fl_replan_sweep_path = save_sweep(
            "nb_3_frozenlake_replan.json", fl_replan_sweep,
        )
        _status = mo.md(
            f"✓ Stale-vs-replan sweep finished in **{_time.time() - _t0:.1f} s** "
            f"— 7 drift levels × 30 episodes × 2 planners. "
            f"Saved to `{fl_replan_sweep_path}`."
        )
    _status
    return (fl_replan_sweep,)


@app.cell
def _(np):
    from numba import njit

    @njit
    def ns_oracle_vi(P, R, gamma):
        """Backward-pass finite-horizon VI on a time-indexed NS-MDP.

        Args:
            P: (T, S, A, S) — P[t, s, a, s'] = Pr(s' | s, a, time t).
            R: (T, S, A, S) — expected reward for the same indexing.
            gamma: discount factor.

        Returns:
            V:      (T+1, S) — V[T] = 0 (terminal); V[t, s] = optimal value.
            policy: (T, S)   — argmax action per (state, time).
        """
        T = P.shape[0]
        S = P.shape[1]
        A = P.shape[2]
        V = np.zeros((T + 1, S))
        policy = np.zeros((T, S), dtype=np.int64)
        for t in range(T - 1, -1, -1):
            for s in range(S):
                best_q = -1e18
                best_a = 0
                for a in range(A):
                    q = 0.0
                    for sp in range(S):
                        p = P[t, s, a, sp]
                        if p > 0.0:
                            q += p * (R[t, s, a, sp] + gamma * V[t + 1, sp])
                    if q > best_q:
                        best_q = q
                        best_a = a
                V[t, s] = best_q
                policy[t, s] = best_a
        return V, policy

    return (ns_oracle_vi,)


@app.cell
def _(np):
    def build_ns_tensors(make_env_fn, T, seed=0):
        """Snapshot the wrapped env's transition table at each of T timesteps.

        The wrapper applies ``update_fn(prob, self.t)`` BEFORE each real
        transition, so the table that governs the agent's t-th step is
        the one *after* update_fn has fired with argument t. We replay
        update_fn(prob, 0), update_fn(prob, 1), ... in-place on a single
        env (no real stepping — avoids agent-state side effects and
        episode truncation) and snapshot after each call.
        """
        env = make_env_fn()
        env.reset(seed=seed)
        n_states = env.unwrapped.observation_space.n
        n_actions = env.unwrapped.action_space.n

        P = np.zeros((T, n_states, n_actions, n_states), dtype=np.float64)
        R = np.zeros((T, n_states, n_actions, n_states), dtype=np.float64)

        for t in range(T):
            env.transition_prob, fired, _ = env.update_fn(env.transition_prob, t)
            if fired:
                env._update_transition_prob_table()
            for s in range(n_states):
                for a in range(n_actions):
                    for prob, sp, reward, _ in env.P[s][a]:
                        P[t, s, a, sp] += float(prob)
                        R[t, s, a, sp] = float(reward)
        return P, R

    return (build_ns_tensors,)


@app.cell
def _(build_ns_tensors, fl_adapt_k, fl_make_env, ns_oracle_vi):
    fl_oracle_horizon = 50
    fl_oracle_gamma = 0.95
    fl_oracle_P, fl_oracle_R = build_ns_tensors(
        lambda: fl_make_env(fl_adapt_k.value), fl_oracle_horizon, seed=0,
    )
    fl_oracle_V, fl_oracle_policy = ns_oracle_vi(
        fl_oracle_P, fl_oracle_R, fl_oracle_gamma,
    )
    return (
        fl_oracle_P,
        fl_oracle_R,
        fl_oracle_V,
        fl_oracle_gamma,
        fl_oracle_horizon,
        fl_oracle_policy,
    )


@app.cell
def _(np):
    from numba import njit as _njit

    @_njit
    def stationary_vi_jit(P, R, gamma, max_iter=2000, theta=1e-6):
        """Full infinite-horizon stationary VI on a single (S, A, S) snapshot.

        This is *not* a one-step lookahead. We iterate the Bellman backup
        until ‖ΔV‖_∞ < theta (or hit max_iter), treating the snapshotted
        (P, R) as if it held forever. Used for the myopic-replan agent:
        at each time t we plug in (P_t, R_t) and solve the full
        stationary Bellman equation, which models an agent that has no
        knowledge of future drift but plans the entire horizon under
        the *most recently observed* model.
        """
        S = P.shape[0]
        A = P.shape[1]
        V = np.zeros(S)
        for _ in range(max_iter):
            delta = 0.0
            for s in range(S):
                v_old = V[s]
                best = -1e18
                for a in range(A):
                    q = 0.0
                    for sp in range(S):
                        p = P[s, a, sp]
                        if p > 0.0:
                            q += p * (R[s, a, sp] + gamma * V[sp])
                    if q > best:
                        best = q
                V[s] = best
                d = abs(v_old - V[s])
                if d > delta:
                    delta = d
            if delta < theta:
                break
        policy = np.zeros(S, dtype=np.int64)
        for s in range(S):
            best_a = 0
            best_q = -1e18
            for a in range(A):
                q = 0.0
                for sp in range(S):
                    p = P[s, a, sp]
                    if p > 0.0:
                        q += p * (R[s, a, sp] + gamma * V[sp])
                if q > best_q:
                    best_q = q
                    best_a = a
            policy[s] = best_a
        return V, policy

    return (stationary_vi_jit,)


@app.cell
def _(np):
    from numba import njit as _njit_pe

    @_njit_pe
    def policy_eval_ns_jit(P, R, gamma, policy):
        """Backward-pass policy evaluation under the true non-stationary tensor.

        V^π(s, t) = expected return from (s, t) when actions are chosen
        by ``policy`` and dynamics are governed by the time-indexed
        ``P``/``R``. No max — we plug in the policy's action and propagate.
        Used for apples-to-apples comparison: each planner's policy is
        evaluated under the *same* (true) dynamics, so panel-to-panel
        differences reflect policy quality alone, not horizon assumptions.

        Args:
            P: (T, S, A, S) — true non-stationary transition tensor.
            R: (T, S, A, S) — true non-stationary reward tensor.
            gamma: discount factor.
            policy: (T, S) int array — action per (time, state). For a
                stale (time-invariant) policy, tile (S,) → (T, S) before
                calling.

        Returns:
            V: (T+1, S) — V[T] = 0; V[t, s] = E[return from (s, t)].
        """
        T = P.shape[0]
        S = P.shape[1]
        V = np.zeros((T + 1, S))
        for t in range(T - 1, -1, -1):
            for s in range(S):
                a = policy[t, s]
                v = 0.0
                for sp in range(S):
                    p = P[t, s, a, sp]
                    if p > 0.0:
                        v += p * (R[t, s, a, sp] + gamma * V[t + 1, sp])
                V[t, s] = v
        return V

    @_njit_pe
    def oracle_q_gap_jit(P, R, gamma):
        """Oracle Q-values and action gap (max Q* − second-best Q*) per (s, t).

        The action gap measures how *decisive* the optimal choice is. Big
        gap → picking the wrong action costs a lot; small gap → multiple
        actions are roughly tied and the policy disagreement is cheap.
        Invariant to horizon scaling, unlike V*.

        Returns:
            Q: (T, S, A) — Q*(s, a, t) under oracle backward induction.
            gap: (T, S) — max_a Q* − second_max_a Q* at each (s, t).
        """
        T = P.shape[0]
        S = P.shape[1]
        A = P.shape[2]
        V = np.zeros((T + 1, S))
        Q = np.zeros((T, S, A))
        for t in range(T - 1, -1, -1):
            for s in range(S):
                best_q = -1e18
                for a in range(A):
                    q = 0.0
                    for sp in range(S):
                        p = P[t, s, a, sp]
                        if p > 0.0:
                            q += p * (R[t, s, a, sp] + gamma * V[t + 1, sp])
                    Q[t, s, a] = q
                    if q > best_q:
                        best_q = q
                V[t, s] = best_q
        gap = np.zeros((T, S))
        for t in range(T):
            for s in range(S):
                best = -1e18
                second = -1e18
                for a in range(A):
                    q = Q[t, s, a]
                    if q > best:
                        second = best
                        best = q
                    elif q > second:
                        second = q
                gap[t, s] = best - second
        return Q, gap

    return oracle_q_gap_jit, policy_eval_ns_jit


@app.cell
def _(FL_ACTION_ARROWS, FL_MAP):
    def fl_draw_value_panel(ax, V_grid, policy_grid, title, vmax):
        """Render a 4x4 V/policy heatmap with hole/goal/arrow annotations."""
        im = ax.imshow(V_grid, cmap="viridis", vmin=0.0, vmax=vmax)
        for r in range(4):
            for c in range(4):
                tile = FL_MAP[r][c]
                if tile == "H":
                    ax.text(c, r, "H", ha="center", va="center",
                            color="red", fontsize=18, fontweight="bold")
                elif tile == "G":
                    ax.text(c, r, "G", ha="center", va="center",
                            color="gold", fontsize=18, fontweight="bold")
                else:
                    ax.text(c, r - 0.18,
                            FL_ACTION_ARROWS[int(policy_grid[r, c])],
                            ha="center", va="center",
                            color="white", fontsize=16)
                ax.text(c, r + 0.32, f"{V_grid[r, c]:.2f}",
                        ha="center", va="center",
                        color="white", fontsize=7)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    return (fl_draw_value_panel,)


@app.cell
def _(FL_ACTION_ARROWS, FL_MAP, np):
    from matplotlib.colors import ListedColormap as _ListedCmap
    from matplotlib.patches import Patch as _Patch

    # One color per action: 0=Left, 1=Down, 2=Right, 3=Up.
    FL_ACTION_COLORS = ["#4c72b0", "#dd8452", "#55a467", "#8172b2"]
    _FL_ACTION_CMAP = _ListedCmap(FL_ACTION_COLORS)
    _FL_ACTION_CMAP.set_bad(color="#dddddd")

    FL_ACTION_LEGEND = [
        _Patch(facecolor=FL_ACTION_COLORS[0], label=f"0  ←  Left"),
        _Patch(facecolor=FL_ACTION_COLORS[1], label=f"1  ↓  Down"),
        _Patch(facecolor=FL_ACTION_COLORS[2], label=f"2  →  Right"),
        _Patch(facecolor=FL_ACTION_COLORS[3], label=f"3  ↑  Up"),
    ]

    def fl_draw_policy_panel(ax, policy_grid, title):
        """4x4 action heatmap, color = action; H/G tiles greyed out."""
        masked = np.array(policy_grid, dtype=float)
        for r in range(4):
            for c in range(4):
                if FL_MAP[r][c] in ("H", "G"):
                    masked[r, c] = np.nan
        masked = np.ma.array(masked, mask=np.isnan(masked))
        im = ax.imshow(masked, cmap=_FL_ACTION_CMAP, vmin=-0.5, vmax=3.5)
        for r in range(4):
            for c in range(4):
                tile = FL_MAP[r][c]
                if tile == "H":
                    ax.text(c, r, "H", ha="center", va="center",
                            color="red", fontsize=16, fontweight="bold")
                elif tile == "G":
                    ax.text(c, r, "G", ha="center", va="center",
                            color="gold", fontsize=16, fontweight="bold")
                else:
                    ax.text(c, r,
                            FL_ACTION_ARROWS[int(policy_grid[r, c])],
                            ha="center", va="center",
                            color="white", fontsize=20)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    return FL_ACTION_COLORS, FL_ACTION_LEGEND, fl_draw_policy_panel


@app.cell
def _(type_mismatch_checker):
    def fl_oracle_rollout(env, oracle_policy, seed=0):
        """Time-aware rollout — `oracle_policy` is a (T, S) int array."""
        obs, _ = env.reset(seed=seed)
        state, _ = type_mismatch_checker(obs)
        done = trunc = False
        t = 0
        ep_return = 0.0
        T = oracle_policy.shape[0]
        while not (done or trunc):
            a = int(oracle_policy[min(t, T - 1), state])
            obs, reward, done, trunc, _ = env.step(a)
            state, reward = type_mismatch_checker(obs, reward)
            ep_return += reward
            t += 1
        return ep_return

    return (fl_oracle_rollout,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Planner comparison — MCTS · PA-MCTS · RATS · DQN on your custom env</div></div>

    ### Pick the env to benchmark on. All three options live on a tabular gridworld where we can build a **time-augmented oracle V tensor**, which is what makes the cumulative Q-gap a meaningful regret signal:

    - **FrozenLake — gallery** — `fl_your_make_env`, the gallery scheduler × drift combination from the dropdowns at the top of Module 1.
    - **Bridge — gallery** — `bridge_make_env` with the per-side drift sliders.
    - **FrozenLake — your registered env** — `fl_register_make_env` from the *"build & register your own NS-MDP"* exercise above. Only usable once that exercise compiles (✓ status); otherwise this option silently falls back to the gallery FrozenLake.

    ### Budget: 3 seeds × 50 steps, MCTS $m{=}30$, RATS depth 3. DQN's contextual weights only fit FrozenLake's $(s, p_\text{slip})$ obs space — it auto-skips on Bridge.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    fl_planners_env = mo.ui.dropdown(
        options=[
            "FrozenLake — gallery",
            "Bridge — gallery",
            "FrozenLake — your registered env",
        ],
        value="FrozenLake — gallery",
        label="Benchmark env",
    )
    fl_planners_run = mo.ui.run_button(label="▶ Run the planner comparison")
    mo.hstack([fl_planners_env, fl_planners_run], justify="start", gap=2)
    return fl_planners_env, fl_planners_run


@app.cell(hide_code=True)
def _(mo):
    from textwrap import dedent as _dedent
    _kwargs_default = _dedent('''\
        # Edit these and re-run. Each planner is rebuilt each step from the
        # live get_planning_env(); the SB3 model is loaded once.

        # MCTS — vanilla UCB1 tree search.
        MCTS_KWARGS = dict(d=20, m=30, c=1.4, gamma=0.95)

        # RATS — risk-averse min-max tree (Lecarpentier & Rachelson, 2019).
        # L_p is the *agent's* Lipschitz bound (RATS' uncertainty ball).
        # Set L_p >= ENV_L below for a faithful match against scenario B.
        RATS_KWARGS = dict(gamma=0.95, max_depth=3, L_p=1.0, L_r=0.0, tau=1.0)

        # Scenario B (RATS Lipschitz drift) — env's Lipschitz constant.
        # Per-step W1(P_t, P_{t+1}) <= ENV_L is enforced by
        # LCBoundedDistrubutionUpdate.
        ENV_L = 0.15

        # PA-MCTS — convex combination of MCTS + SB3 DDQN prior.
        #   alpha = 0   → pure MCTS
        #   alpha = 1   → pure SB3 prior
        #   alpha = 0.5 → balanced (default)
        PAMCTS_ALPHA = 0.5
        PAMCTS_MCTS_KWARGS = dict(d=20, m=20, c=1.4, gamma=0.95)

        # SB3 weights to load for DQN + PA-MCTS prior.
        # FrozenLake-only — skipped on Bridge.
        SB3_DQN_PATH = "model_weights/contextual_ddqn_frozenlake.zip"

        # Rollout budget (small = fast, larger = smoother curves).
        # 20 seeds gives stable means without making each scenario take
        # forever; drop to 5-10 if you're iterating on the env itself.
        N_SEEDS   = 20
        MAX_STEPS = 50
        ''')
    fl_planners_kwargs = mo.ui.code_editor(
        value=_kwargs_default, language="python", min_height=18,
    )
    mo.vstack([
        mo.md(
            "**Knobs.** Edit kwargs inline below — MCTS budget, RATS "
            "depth/Lipschitz, PA-MCTS α, SB3 weights path, seed count. "
            "Re-run the button above to see the new curves."
        ),
        fl_planners_kwargs,
    ])
    return (fl_planners_kwargs,)


@app.cell
def _(
    bridge_P,
    bridge_R,
    bridge_make_env,
    bridge_oracle_V,
    bridge_stationary_policy,
    build_ns_tensors,
    fl_adapt_k,
    fl_oracle_P,
    fl_oracle_R,
    fl_oracle_V,
    fl_oracle_gamma,
    fl_oracle_horizon,
    fl_planners_env,
    fl_planners_kwargs,
    fl_planners_run,
    fl_register_make_env,
    fl_register_ok,
    fl_stationary_policy,
    fl_your_make_env,
    mo,
    np,
    ns_oracle_vi,
    plt,
    type_mismatch_checker,
):
    import time as _time
    if not fl_planners_run.value:
        _out = mo.md(
            "_Choose an env, edit kwargs above if you want, then click run. "
            "~30 s on a laptop._"
        )
    else:
        _t0 = _time.time()
        from ns_gym.benchmark_algorithms import MCTS as _MCTS
        try:
            from ns_gym.benchmark_algorithms import RATS as _RATS
        except Exception:
            _RATS = None

        # Parse user kwargs (fall back to defaults on error).
        _kw = {
            "MCTS_KWARGS": dict(d=20, m=30, c=1.4, gamma=0.95),
            "RATS_KWARGS": dict(gamma=0.95, max_depth=3,
                                L_p=1.0, L_r=0.0, tau=1.0),
            "PAMCTS_ALPHA": 0.5,
            "PAMCTS_MCTS_KWARGS": dict(d=20, m=20, c=1.4, gamma=0.95),
            "SB3_DQN_PATH": "model_weights/contextual_ddqn_frozenlake.zip",
            "N_SEEDS": 20,
            "MAX_STEPS": 50,
        }
        _kw_err = None
        try:
            exec(fl_planners_kwargs.value, _kw)
        except Exception as _e:
            _kw_err = f"{type(_e).__name__}: {_e}"

        _env_choice = fl_planners_env.value
        _is_bridge = _env_choice.startswith("Bridge")
        _is_registered = _env_choice.startswith("FrozenLake — your registered")
        _env_source_note = ""

        if _is_bridge:
            _make_env = bridge_make_env
            _P_t, _R_t, _V_t = bridge_P, bridge_R, bridge_oracle_V
            _stale_pi = bridge_stationary_policy
        elif _is_registered:
            # User picked their registered NS-MDP from the exercise above.
            # If it didn't compile, fall back silently to the gallery
            # FrozenLake so the panel still produces a result.
            if fl_register_ok and fl_register_make_env is not None:
                try:
                    _make_env = fl_register_make_env
                    _P_t, _R_t = build_ns_tensors(
                        _make_env, fl_oracle_horizon, seed=0,
                    )
                    _V_t, _ = ns_oracle_vi(_P_t, _R_t, fl_oracle_gamma)
                    _env_source_note = (
                        " (using your registered env from the exercise above)"
                    )
                except Exception as _e:
                    _kw_err = (
                        f"{type(_e).__name__} while building oracle "
                        f"tensor on your registered env: {_e}"
                    )
                    _make_env = fl_your_make_env
                    _P_t, _R_t, _V_t = fl_oracle_P, fl_oracle_R, fl_oracle_V
                    _env_source_note = (
                        " (your registered env errored — fell back to gallery)"
                    )
            else:
                _make_env = fl_your_make_env
                _P_t, _R_t, _V_t = fl_oracle_P, fl_oracle_R, fl_oracle_V
                _env_source_note = (
                    " (registered exercise not compiled — fell back to gallery)"
                )
            _stale_pi = fl_stationary_policy
        else:
            _make_env = fl_your_make_env
            _P_t, _R_t, _V_t = fl_oracle_P, fl_oracle_R, fl_oracle_V
            _stale_pi = fl_stationary_policy

        # Load SB3 DQN once (FrozenLake only — Bridge has no trained weights).
        _dqn_model = None
        if not _is_bridge:
            try:
                from stable_baselines3 import DQN as _SB3_DQN
                _dqn_model = _SB3_DQN.load(_kw["SB3_DQN_PATH"])
            except Exception:
                _dqn_model = None

        def _oracle_q(t_idx, state):
            n_a = _P_t.shape[2]
            n_s = _P_t.shape[1]
            return np.array([
                sum(
                    _P_t[t_idx, state, a, sp]
                    * (_R_t[t_idx, state, a, sp]
                       + fl_oracle_gamma * _V_t[t_idx + 1, sp])
                    for sp in range(n_s)
                )
                for a in range(n_a)
            ])

        def _sb3_obs_vec(state, env):
            """Build the 19-d (one-hot state + 3-d slip dist) obs the
            contextual DDQN was trained on."""
            onehot = np.eye(16, dtype=np.float32)[int(state)]
            pvec = np.asarray(env.transition_prob, dtype=np.float32)
            if pvec.size != 3:
                pvec = np.array(
                    [pvec[0], (1 - pvec[0]) / 2, (1 - pvec[0]) / 2],
                    dtype=np.float32,
                )
            return np.concatenate([onehot, pvec])

        def _sb3_q_values(obs_vec):
            """Forward through SB3's q_net to get raw action values."""
            import torch
            with torch.no_grad():
                t = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
                q = _dqn_model.q_net(t).cpu().numpy().ravel()
            return q

        def _pick_stale(state, _t, _env):
            return int(_stale_pi[state])

        def _pick_mcts(state, _t, env):
            penv = env.get_planning_env()
            agent = _MCTS(penv, state=int(state), **_kw["MCTS_KWARGS"])
            a, _ = agent.search()
            return int(a)

        def _make_rats_picker():
            if _RATS is None:
                return None
            # Inject a private hole=-1 view of the planning env so RATS'
            # depth-2 worst-case minimax has a usable gradient. Without
            # this, every leaf returns 0 reward (goal too far) and RATS
            # picks arbitrary actions across all contexts -- same fix
            # used in scenarios B and C.
            def _inject_hole_penalty(penv):
                P = penv.unwrapped.P
                if not isinstance(P, dict):
                    return penv  # Bridge envs use a different P shape
                new_P = {
                    s: {
                        a: [
                            (p, ns, -1.0 if (d and r <= 0) else float(r), d)
                            for p, ns, r, d in P[s][a]
                        ] for a in P[s]
                    } for s in P
                }
                penv.P = new_P
                penv.unwrapped.P = new_P
                if hasattr(penv, "intial_p"):
                    penv.intial_p = new_P
                return penv
            def _pick(state, _t, env):
                penv = _inject_hole_penalty(env.get_planning_env())
                agent = _RATS(
                    action_space=penv.action_space, **_kw["RATS_KWARGS"],
                )
                return int(agent.act(observation=int(state), env=penv))
            return _pick

        def _make_dqn_picker():
            if _dqn_model is None:
                return None
            def _pick(state, _t, env):
                obs_vec = _sb3_obs_vec(state, env)
                action, _ = _dqn_model.predict(obs_vec, deterministic=True)
                return int(action)
            return _pick

        def _make_pamcts_picker():
            """Manual PA-MCTS — convex combo of MCTS Q-values and SB3 q_net.
            Mirrors ns_gym.benchmark_algorithms.PAMCTS.search() but bypasses
            its DDQN_agent constructor (which expects raw .pth). We keep
            the SB3 model directly."""
            if _dqn_model is None:
                return None
            _alpha = float(_kw["PAMCTS_ALPHA"])

            def _pick(state, _t, env):
                penv = env.get_planning_env()
                # MCTS branch
                mcts_agent = _MCTS(
                    penv, state=int(state), **_kw["PAMCTS_MCTS_KWARGS"],
                )
                _, mcts_q = mcts_agent.search()
                mcts_q = np.asarray(mcts_q, dtype=np.float32)
                # DDQN branch (SB3 q_net forward pass)
                obs_vec = _sb3_obs_vec(state, env)
                ddqn_q = _sb3_q_values(obs_vec).astype(np.float32)
                # Both normalized to [0, 1] (matches PAMCTS source)
                eps = 1e-8
                mcts_n = (mcts_q - mcts_q.min()) / (
                    mcts_q.max() - mcts_q.min() + eps
                )
                ddqn_n = (ddqn_q - ddqn_q.min()) / (
                    ddqn_q.max() - ddqn_q.min() + eps
                )
                hybrid = _alpha * ddqn_n + (1.0 - _alpha) * mcts_n
                return int(np.argmax(hybrid))
            return _pick

        T_ORACLE = _V_t.shape[0] - 1

        # Oracle VI = ceiling baseline. Argmax over the time-indexed
        # oracle Q tensor; effectively free per decision.
        def _pick_oracle(state, t, _env):
            return int(np.argmax(_oracle_q(min(t, T_ORACLE - 1), int(state))))

        _planners = {
            "Oracle VI":  _pick_oracle,
            "stale-VI":   _pick_stale,
            "MCTS":       _pick_mcts,
            "PA-MCTS":    _make_pamcts_picker(),
            "RATS":       _make_rats_picker(),
            "DQN (SB3)":  _make_dqn_picker(),
        }

        MAX_STEPS = int(_kw["MAX_STEPS"])
        SEEDS = list(range(int(_kw["N_SEEDS"])))

        def _rollout(picker):
            """Returns (final_returns, final_qgaps) -- one entry per seed."""
            final_returns = []
            final_qgaps = []
            for s in SEEDS:
                env = _make_env()
                obs, _ = env.reset(seed=s)
                state, _ = type_mismatch_checker(obs)
                np.random.seed(s)
                cum_r, cum_q = 0.0, 0.0
                done = trunc = False
                t = 0
                while not (done or trunc) and t < MAX_STEPS:
                    t_idx = min(t, T_ORACLE - 1)
                    qa = _oracle_q(t_idx, int(state))
                    a_oracle = int(np.argmax(qa))
                    a_taken = int(picker(int(state), t, env))
                    cum_q += float(qa[a_oracle] - qa[a_taken])
                    obs, reward, done, trunc, _ = env.step(a_taken)
                    state, reward = type_mismatch_checker(obs, reward)
                    cum_r += float(reward)
                    t += 1
                final_returns.append(cum_r)
                final_qgaps.append(cum_q)
            return np.asarray(final_returns), np.asarray(final_qgaps)

        _traces = {}
        _errors = {}
        if _kw_err:
            _errors["⚙️ kwargs"] = _kw_err
        _planner_items = list(_planners.items())
        for _name, _pick in mo.status.progress_bar(
            _planner_items,
            title="Planner comparison",
            subtitle="rolling out each planner across seeds…",
            remove_on_exit=True,
        ):
            if _pick is None:
                if _is_bridge and _name in ("DQN (SB3)", "PA-MCTS"):
                    _errors[_name] = (
                        "no SB3 weights for Bridge (FrozenLake-only model)"
                    )
                else:
                    _errors[_name] = "unavailable (skipped)"
                continue
            try:
                _traces[_name] = _rollout(_pick)
            except Exception as _e:
                _errors[_name] = f"{type(_e).__name__}: {_e}"

        _colors = {
            "Oracle VI":  "#1a1a1a",
            "stale-VI":   "tab:red",
            "MCTS":       "tab:blue",
            "PA-MCTS":    "tab:orange",
            "RATS":       "tab:purple",
            "DQN (SB3)":  "tab:green",
        }
        # Two side-by-side violins: per-seed final returns (left) and
        # per-seed final Q-gaps (right). The line plots used to live here
        # but a) cumulative reward on terminal-only-reward envs is mostly
        # flat with one late jump, and b) cumulative Q-gap is monotone
        # non-decreasing -- the *final* number per seed carries the
        # comparison, and the violin shows the spread the line plot hid.
        _env_label = "Bridge" if _is_bridge else "FrozenLake"
        _fig, (_ax_r, _ax_q) = plt.subplots(
            1, 2, figsize=(13, 4.6), layout="constrained",
        )
        _names = list(_traces.keys())
        _positions = list(range(len(_names)))
        _rng = np.random.default_rng(0)
        _stagger_y_r = (1.10, 1.28)

        # ---- final-return violin (LEFT) ----
        _ret_data = [_traces[n][0] for n in _names]
        if _ret_data:
            _vp = _ax_r.violinplot(
                _ret_data, positions=_positions, showmeans=False,
                showmedians=False, showextrema=False, widths=0.75,
            )
            for _body, _n in zip(_vp["bodies"], _names):
                _body.set_facecolor(_colors.get(_n, "gray"))
                _body.set_edgecolor("black")
                _body.set_alpha(0.45)
                _body.set_linewidth(0.6)
        for _i, _n in enumerate(_names):
            _arr = _traces[_n][0]
            if len(_arr) == 0:
                continue
            _xs = _i + _rng.uniform(-0.12, 0.12, size=len(_arr))
            _ax_r.scatter(_xs, _arr, s=28, alpha=0.75,
                          color=_colors.get(_n, "gray"),
                          edgecolor="black", linewidth=0.4, zorder=4)
            _mu = float(np.mean(_arr))
            _se = float(np.std(_arr, ddof=1) / np.sqrt(len(_arr))) if len(_arr) > 1 else 0.0
            _ax_r.errorbar(_i, _mu, yerr=_se, fmt="D", color="white",
                           markerfacecolor="black", markeredgecolor="black",
                           markersize=7, capsize=5, zorder=5)
            _wins = int(np.sum(_arr > 0))
            _y_text = _stagger_y_r[_i % 2]
            _ax_r.text(_i, _y_text,
                       f"μ={_mu:.2f}±{_se:.02f}\n{_wins}/{len(_arr)} goals",
                       ha="center", va="bottom", fontsize=8)
        _ax_r.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
        _ax_r.axhline(1, color="grey", linewidth=0.5, alpha=0.4)
        _ax_r.set_xticks(_positions)
        _ax_r.set_xticklabels(_names, fontsize=8, rotation=30, ha="right")
        _ax_r.set_ylabel("return per seed (1 = goal reached)")
        _ax_r.set_ylim(-0.15 if not _is_bridge else -1.15, 1.65)
        _ax_r.set_title(
            f"Per-seed return  •  {_env_label}  •  k={fl_adapt_k.value:.2f}",
            fontsize=10,
        )

        # ---- final-Q-gap violin (RIGHT) ----
        _gap_data = [_traces[n][1] for n in _names]
        if _gap_data:
            _vp_q = _ax_q.violinplot(
                _gap_data, positions=_positions, showmeans=False,
                showmedians=False, showextrema=False, widths=0.75,
            )
            for _body, _n in zip(_vp_q["bodies"], _names):
                _body.set_facecolor(_colors.get(_n, "gray"))
                _body.set_edgecolor("black")
                _body.set_alpha(0.45)
                _body.set_linewidth(0.6)
        # y-range for staggered annotations: depends on the data, so
        # compute after the data is drawn
        _gap_max = float(max((np.max(g) for g in _gap_data if len(g) > 0), default=1.0))
        _gap_min = float(min((np.min(g) for g in _gap_data if len(g) > 0), default=0.0))
        _gap_span = max(_gap_max - _gap_min, 1e-3)
        _stagger_y_q = (_gap_max + 0.05 * _gap_span, _gap_max + 0.18 * _gap_span)
        for _i, _n in enumerate(_names):
            _arr = _traces[_n][1]
            if len(_arr) == 0:
                continue
            _xs = _i + _rng.uniform(-0.12, 0.12, size=len(_arr))
            _ax_q.scatter(_xs, _arr, s=28, alpha=0.75,
                          color=_colors.get(_n, "gray"),
                          edgecolor="black", linewidth=0.4, zorder=4)
            _mu = float(np.mean(_arr))
            _se = float(np.std(_arr, ddof=1) / np.sqrt(len(_arr))) if len(_arr) > 1 else 0.0
            _ax_q.errorbar(_i, _mu, yerr=_se, fmt="D", color="white",
                           markerfacecolor="black", markeredgecolor="black",
                           markersize=7, capsize=5, zorder=5)
            _y_text = _stagger_y_q[_i % 2]
            _ax_q.text(_i, _y_text, f"μ={_mu:.2f}±{_se:.02f}",
                       ha="center", va="bottom", fontsize=8)
        _ax_q.axhline(0, color="grey", linewidth=0.6)
        _ax_q.set_xticks(_positions)
        _ax_q.set_xticklabels(_names, fontsize=8, rotation=30, ha="right")
        _ax_q.set_ylabel(r"final cumulative $\Delta_Q$ (regret vs. oracle)")
        _ax_q.set_ylim(min(_gap_min - 0.05 * _gap_span, -0.05),
                        _gap_max + 0.35 * _gap_span)
        _ax_q.set_title(
            f"Per-seed Q-gap  •  {_env_label}  •  lower is better",
            fontsize=10,
        )

        _err_md = ""
        if _errors:
            _err_md = "\n\n_Skipped:_  " + "  ·  ".join(
                f"**{n}** ({why})" for n, why in _errors.items()
            )
        _elapsed = _time.time() - _t0
        _ran = len(_traces)
        _done_md = mo.md(
            f"✓ Planner comparison finished in **{_elapsed:.1f} s** — "
            f"{_ran} planner(s) rolled out × {len(SEEDS)} seeds × "
            f"{MAX_STEPS} steps{_env_source_note}." + _err_md
        )
        _out = mo.vstack([_fig, _done_md])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #5a3a8a; background: #f3eef7;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">Where each algorithm earns its keep</div></div>

    ### The gallery comparison above runs everyone on the *same* env — useful as a baseline, but it doesn't tell you *when* you'd reach for each method in the wild. Below, three crafted scenarios designed to play to one algorithm's strengths each:

    1. **PA-MCTS scenario** — pretrained policy, *one big notified
       ### shock*, no time to retrain. PA-MCTS combines the prior with a corrected lookahead.
    2. **RATS scenario** — bounded but sharp drift (sigmoid cliff).
       ### RATS' min-max search is conservative on purpose — when reality turns adversarial it survives where MCTS gambles.
    3. **DDQN scenario** — *decision-time* box plots, not return.
       ### Tree search is fast on FrozenLake and slow on continuous control; a learned policy is constant-time everywhere.

    ### Each scenario has its own run-button and uses the kwargs editor above so you can tune budgets and re-run.
    """)
    return


@app.cell
def _(np, ns_oracle_vi, plt):
    import time as _time_mod

    _PLANNER_COLORS = {
        "Oracle VI":     "#1a1a1a",  # near-black, signals "ceiling"
        "stale-VI":      "tab:red",
        "MCTS":          "tab:blue",
        "PA-MCTS":       "tab:orange",
        "PA-MCTS α=0.25": "#ffbb78",  # light orange
        "PA-MCTS α=0.50": "tab:orange",
        "PA-MCTS α=0.75": "#d97706",  # dark orange
        "RATS":          "tab:purple",
        "DQN (SB3)":     "tab:green",
    }

    def _scenario_default_kwargs():
        return {
            "MCTS_KWARGS": dict(d=20, m=30, c=1.4, gamma=0.95),
            "RATS_KWARGS": dict(gamma=0.95, max_depth=3,
                                L_p=1.0, L_r=0.0, tau=1.0),
            "PAMCTS_ALPHA": 0.5,
            "PAMCTS_MCTS_KWARGS": dict(d=20, m=20, c=1.4, gamma=0.95),
            "SB3_DQN_PATH": "model_weights/contextual_ddqn_frozenlake.zip",
            "N_SEEDS": 20,
            "ENV_L": 0.15,
            "ORACLE_GAMMA": 0.95,
        }

    def parse_scenario_kwargs(editor_value):
        kw = _scenario_default_kwargs()
        try:
            exec(editor_value, kw)
        except Exception:
            pass
        return kw

    def try_load_sb3(path):
        try:
            from stable_baselines3 import DQN as _SB3_DQN
            return _SB3_DQN.load(path)
        except Exception:
            return None

    def build_full_pickers(kwargs, dqn_model, stale_policy, MCTS_cls, RATS_cls):
        """Returns dict: name -> pick(state, t, env). All five planners."""
        def _sb3_obs_vec(state, env):
            onehot = np.eye(16, dtype=np.float32)[int(state)]
            pvec = np.asarray(env.transition_prob, dtype=np.float32)
            if pvec.size != 3:
                pvec = np.array(
                    [pvec[0], (1 - pvec[0]) / 2, (1 - pvec[0]) / 2],
                    dtype=np.float32,
                )
            return np.concatenate([onehot, pvec])

        def stale_pick(state, _t, _env):
            if stale_policy is None:
                raise RuntimeError("stale_policy not provided")
            return int(stale_policy[state])

        def mcts_pick(state, _t, env):
            penv = env.get_planning_env()
            agent = MCTS_cls(penv, state=int(state), **kwargs["MCTS_KWARGS"])
            a, _ = agent.search()
            return int(a)

        def _rats_inject_hole_penalty(penv):
            """Replace transitions of the form (p, ns, 0, done=True) --
            i.e. landing on a hole -- with (p, ns, -1, done=True) so
            RATS' shallow worst-case minimax has a usable gradient.
            FrozenLake-style P (dict-of-(prob, ns, r, done) tuples)
            only -- non-dict P (e.g. Bridge slip-vector) is left alone."""
            P = penv.unwrapped.P
            if not isinstance(P, dict):
                return penv
            new_P = {
                s: {
                    a: [
                        (p, ns, -1.0 if (d and r <= 0) else float(r), d)
                        for p, ns, r, d in P[s][a]
                    ] for a in P[s]
                } for s in P
            }
            penv.P = new_P
            penv.unwrapped.P = new_P
            if hasattr(penv, "intial_p"):
                penv.intial_p = new_P
            return penv

        def rats_pick(state, _t, env):
            if RATS_cls is None:
                raise RuntimeError("RATS unavailable")
            penv = _rats_inject_hole_penalty(env.get_planning_env())
            return int(RATS_cls(
                action_space=penv.action_space, **kwargs["RATS_KWARGS"],
            ).act(observation=int(state), env=penv))

        def dqn_pick(state, _t, env):
            if dqn_model is None:
                raise RuntimeError("SB3 model not loaded")
            obs_vec = _sb3_obs_vec(state, env)
            action, _ = dqn_model.predict(obs_vec, deterministic=True)
            return int(action)

        def make_pamcts_pick(alpha):
            """Return a PA-MCTS picker bound to the given α. Uses
            the canonical ``ns_gym.benchmark_algorithms.PAMCTS`` class
            via its new ``q_value_fn`` kwarg, which lets us plug in the
            SB3 contextual DDQN through ``StableBaselineWrapper``
            without converting weights to a raw ``.pth`` state dict.
            Same MCTS configuration as the standalone MCTS planner --
            only the DDQN-prior mixing weight α differs."""
            if dqn_model is None:
                return None
            from ns_gym.base import StableBaselineWrapper
            from ns_gym.benchmark_algorithms import PAMCTS
            mcts_kw = kwargs["MCTS_KWARGS"]
            wrap = StableBaselineWrapper(
                dqn_model,
                obs_fn=lambda state, env: _sb3_obs_vec(state, env),
            )
            agent = PAMCTS(
                alpha=alpha,
                mcts_iter=mcts_kw["m"],
                mcts_search_depth=mcts_kw["d"],
                mcts_discount_factor=mcts_kw["gamma"],
                mcts_exploration_constant=mcts_kw["c"],
                state_space_size=16, action_space_size=4,
                q_value_fn=wrap.q_values,
            )
            def _pick(state, _t, env):
                return int(agent.act(int(state), env.get_planning_env()))
            return _pick

        return {
            "stale-VI":  stale_pick,
            "MCTS":      mcts_pick,
            "PA-MCTS":   make_pamcts_pick(float(kwargs["PAMCTS_ALPHA"])),
            "RATS":      rats_pick,
            "DQN (SB3)": dqn_pick,
            # The factory used by scenario A's α-sweep.
            "_make_pamcts_pick": make_pamcts_pick,
        }

    def run_scenario_metrics(
        env_factory, planners, oracle_P, oracle_R, oracle_V,
        gamma, n_seeds, max_steps, type_mismatch_checker,
        progress_wrap=None,
    ):
        """Run every planner; capture per-step cum reward, cum Q-gap,
        decision times, and per-seed final returns. Returns (metrics, errors).

        Oracle VI is auto-prepended as the ceiling baseline. Crucially,
        for envs whose drift sequence is stochastic across seeds (e.g.
        ``LCBoundedDistrubutionUpdate``), the ceiling is *rebuilt per
        seed* using that seed's actual drift trajectory. Otherwise the
        precomputed-once oracle (built for seed 0) under-represents the
        true optimum on later seeds, and a robust planner like RATS can
        appear to beat it -- which is impossible by construction.
        """

        def _seed_env_inner_rngs(env, sd):
            """Reseed any RNG-bearing update functions on env so each
            seed gets a reproducible drift sequence and every planner
            sees the SAME drift per seed (fair head-to-head). No-op for
            envs whose update fns don't carry an RNG."""
            try:
                upd = getattr(env, "update_fn", None)
                if upd is None:
                    return
                if hasattr(upd, "seed"):
                    upd.seed(sd)
                inner = getattr(upd, "update_fn", None)
                if inner is not None and hasattr(inner, "seed"):
                    inner.seed(sd)
            except Exception:
                pass

        def _build_oracle_for_seed(sd):
            """Snapshot the env's drift trajectory under seed `sd` and run
            VI on the resulting time-indexed (P, R) tensors. This is the
            *true* optimal policy for the drift this seed will actually
            experience -- not the precomputed seed-0 oracle."""
            try:
                e = env_factory()
                e.reset(seed=sd)
                _seed_env_inner_rngs(e, sd)
                n_states = e.unwrapped.observation_space.n
                n_actions = e.unwrapped.action_space.n
                P_t = np.zeros(
                    (max_steps, n_states, n_actions, n_states), dtype=np.float64,
                )
                R_t = np.zeros(
                    (max_steps, n_states, n_actions, n_states), dtype=np.float64,
                )
                for t in range(max_steps):
                    e.transition_prob, fired, _ = e.update_fn(
                        e.transition_prob, t,
                    )
                    if fired:
                        e._update_transition_prob_table()
                    for s in range(n_states):
                        for a in range(n_actions):
                            for prob, sp, reward, _ in e.P[s][a]:
                                P_t[t, s, a, sp] += float(prob)
                                R_t[t, s, a, sp] = float(reward)
                V_t, _ = ns_oracle_vi(P_t, R_t, gamma)
                return P_t, R_t, V_t
            except Exception:
                # Env doesn't expose update_fn / P / etc. in the form we
                # need -- fall back to the precomputed once-built oracle.
                return oracle_P, oracle_R, oracle_V

        # Cache per-seed oracle tensors -- shared across all planners
        # since they only depend on the env's drift, not the planner.
        _seed_oracles = {}

        def _get_seed_oracle(sd):
            if sd not in _seed_oracles:
                _seed_oracles[sd] = _build_oracle_for_seed(sd)
            return _seed_oracles[sd]

        # "Active oracle" mutated by the per-seed loop and consulted by
        # both _oracle_q (for Q-gap) and _oracle_pick (for Oracle VI).
        _active = {"P": oracle_P, "R": oracle_R, "V": oracle_V}

        def _oracle_q(t_idx, state):
            P_a, R_a, V_a = _active["P"], _active["R"], _active["V"]
            n_a = P_a.shape[2]
            n_s = P_a.shape[1]
            return np.array([
                sum(
                    P_a[t_idx, state, a, sp]
                    * (R_a[t_idx, state, a, sp] + gamma * V_a[t_idx + 1, sp])
                    for sp in range(n_s)
                )
                for a in range(n_a)
            ])

        def _oracle_pick(state, t, _env):
            T_oracle = _active["V"].shape[0] - 1
            return int(np.argmax(_oracle_q(min(t, T_oracle - 1), int(state))))

        planners = {"Oracle VI": _oracle_pick, **planners}

        metrics, errors = {}, {}
        _planner_items = list(planners.items())
        _iter = (
            progress_wrap(_planner_items) if progress_wrap is not None
            else _planner_items
        )

        for name, pick in _iter:
            try:
                cum_r_acc = np.zeros(max_steps, dtype=np.float64)
                cum_q_acc = np.zeros(max_steps, dtype=np.float64)
                param_acc = np.zeros(max_steps, dtype=np.float64)
                n_acc = np.zeros(max_steps, dtype=np.int32)
                decision_times_us = []
                final_returns = []  # one per seed, for the violin plot

                for s in range(n_seeds):
                    # Activate THIS seed's oracle tensors so the Oracle
                    # VI picker and the Q-gap computation both reflect
                    # the optimal policy for the drift this seed will
                    # actually experience.
                    P_s, R_s, V_s = _get_seed_oracle(s)
                    _active["P"], _active["R"], _active["V"] = P_s, R_s, V_s
                    T_oracle_s = V_s.shape[0] - 1

                    env = env_factory()
                    obs, _ = env.reset(seed=s)
                    _seed_env_inner_rngs(env, s)
                    state, _ = type_mismatch_checker(obs)
                    np.random.seed(s)
                    cum_r, cum_q = 0.0, 0.0
                    last_param = float(env.transition_prob[0])
                    done = trunc = False
                    t = 0
                    while not (done or trunc) and t < max_steps:
                        t_idx = min(t, T_oracle_s - 1)
                        qa = _oracle_q(t_idx, int(state))
                        a_oracle = int(np.argmax(qa))

                        _t0 = _time_mod.perf_counter()
                        a_taken = int(pick(state, t, env))
                        decision_times_us.append(
                            (_time_mod.perf_counter() - _t0) * 1e6
                        )

                        cum_q += float(qa[a_oracle] - qa[a_taken])
                        last_param = float(env.transition_prob[0])

                        obs, reward, done, trunc, _ = env.step(a_taken)
                        state, reward = type_mismatch_checker(obs, reward)
                        cum_r += float(reward)

                        cum_r_acc[t] += cum_r
                        cum_q_acc[t] += cum_q
                        param_acc[t] += last_param
                        n_acc[t] += 1
                        t += 1
                    while t < max_steps:
                        cum_r_acc[t] += cum_r
                        cum_q_acc[t] += cum_q
                        param_acc[t] += last_param
                        n_acc[t] += 1
                        t += 1
                    final_returns.append(float(cum_r))

                metrics[name] = {
                    "cum_reward":     cum_r_acc / np.maximum(n_acc, 1),
                    "cum_q_gap":      cum_q_acc / np.maximum(n_acc, 1),
                    "param_trace":    param_acc / np.maximum(n_acc, 1),
                    "decision_times": decision_times_us,
                    "final_returns":  np.asarray(final_returns),
                }
            except Exception as e:
                errors[name] = f"{type(e).__name__}: {e}"

        return metrics, errors

    def plot_scenario_panel(
        metrics, errors, title, ylim_reward=None, plt_mod=plt,
    ):
        """1x2 panel:
            (L) per-seed return distribution (violin + strip + mean ± SE)
            (R) per-step decision time (log-scale box)

        The parameter-trace subplot was removed -- with seeded LC drift
        every planner sees the *same* drift sequence per seed, so all
        the per-method param traces overlapped exactly and added nothing.
        The cumulative-reward-over-time subplot was also dropped; it's
        uninformative for terminal-only-reward envs (FrozenLake's binary
        0/1, CartPole's survival counter) where the curve is a flat line
        with a single late jump. The violin carries the comparison.

        ``ylim_reward`` is accepted for back-compat but unused.
        """
        del ylim_reward
        fig, (ax_v, ax_d) = plt_mod.subplots(
            1, 2, figsize=(13, 5.0), layout="constrained",
            gridspec_kw={"width_ratios": [1.15, 1.0]},
        )

        # ----- per-seed return distribution (violin + strip overlay) -----
        names = list(metrics.keys())
        positions = list(range(len(names)))
        n_seeds_max = 0
        if names:
            returns_per_method = [
                np.asarray(metrics[n].get("final_returns", []))
                for n in names
            ]
            n_seeds_max = max((len(r) for r in returns_per_method), default=0)
            non_empty = [
                (i, r) for i, r in enumerate(returns_per_method) if len(r) > 0
            ]
            if non_empty:
                vp = ax_v.violinplot(
                    [r for _, r in non_empty],
                    positions=[i for i, _ in non_empty],
                    showmeans=False, showmedians=False, showextrema=False,
                    widths=0.75,
                )
                for body, (i, _) in zip(vp["bodies"], non_empty):
                    body.set_facecolor(_PLANNER_COLORS.get(names[i], "gray"))
                    body.set_edgecolor("black")
                    body.set_alpha(0.45)
                    body.set_linewidth(0.6)
            rng = np.random.default_rng(0)
            # Stagger the per-method annotations vertically so they don't
            # overlap when planner names are wide (e.g. "PA-MCTS α=0.25").
            stagger_y = (1.10, 1.28)
            for i, name in enumerate(names):
                r = returns_per_method[i]
                if len(r) == 0:
                    continue
                color = _PLANNER_COLORS.get(name, "gray")
                xs = i + rng.uniform(-0.12, 0.12, size=len(r))
                ax_v.scatter(
                    xs, r, s=28, alpha=0.75, color=color,
                    edgecolor="black", linewidth=0.4, zorder=4,
                )
                mu = float(np.mean(r))
                se = float(np.std(r, ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0
                ax_v.errorbar(
                    i, mu, yerr=se, fmt="D", color="white",
                    markerfacecolor="black", markeredgecolor="black",
                    markersize=7, capsize=5, zorder=5,
                )
                wins = int(np.sum(np.asarray(r) > 0))
                y_text = stagger_y[i % 2]
                ax_v.text(
                    i, y_text,
                    f"μ={mu:.2f}±{se:.02f}\n{wins}/{len(r)} goals",
                    ha="center", va="bottom", fontsize=8,
                )
        ax_v.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
        ax_v.axhline(1, color="grey", linewidth=0.5, alpha=0.4)
        ax_v.set_xticks(positions)
        ax_v.set_xticklabels(names, fontsize=8, rotation=30, ha="right")
        ax_v.set_ylabel("return per seed (1 = goal reached)")
        ax_v.set_ylim(-0.15, 1.65)
        ax_v.set_title(
            f"Per-seed return distribution  (N={n_seeds_max})",
            fontsize=10,
        )

        # ------------------- decision-time box plot ----------------------
        if names:
            data = [metrics[n]["decision_times"] for n in names]
            bp = ax_d.boxplot(
                data, tick_labels=names, patch_artist=True,
                widths=0.55, showfliers=False,
            )
            for patch, n in zip(bp["boxes"], names):
                patch.set_facecolor(_PLANNER_COLORS.get(n, "gray"))
                patch.set_alpha(0.55)
            ax_d.set_yscale("log")
        ax_d.set_ylabel("decision time per step (μs, log)")
        ax_d.set_title("Per-step decision time", fontsize=10)
        ax_d.tick_params(axis="x", rotation=30)
        for label in ax_d.get_xticklabels():
            label.set_horizontalalignment("right")
        ax_d.grid(axis="y", which="both", alpha=0.25)

        fig.suptitle(title, fontsize=11, fontweight=600)
        return fig

    return (
        build_full_pickers,
        parse_scenario_kwargs,
        plot_scenario_panel,
        run_scenario_metrics,
        try_load_sb3,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">A · PA-MCTS — single notified shock + stale prior</div></div>

    ### The setup

    | Component | Choice | Why |
    | :-- | :-- | :-- |
    | **Env** | `FrozenLake-v1` (4×4, `is_slippery=False`) | Same grid as scenarios B/C; the choice that varies is the *temporal* dynamics. |
    | **Initial slip** | `[0.45, 0.275, 0.275]` | Just *below* the contextual DDQN's training band of `Uniform[0.5, 1.0]`, so even the prior is mildly OOD pre-shock. |
    | **Shock** | At step 1, `DistributionStepWiseUpdate` fires once and bumps `P[intended]` up to **0.55**. | Lands the env just *inside* the DDQN's training distribution -- the DDQN's post-shock Q-values are now *partially-calibrated*, the regime where mixing with MCTS pays. |
    | **Notification** | `change_notification=True`, `delta_change_notification=True` | Every planner gets full visibility into the post-shock model via `env.get_planning_env()` -- nothing is hidden; the comparison is purely about how each method *uses* the information. |
    | **DDQN** | `model_weights/contextual_ddqn_frozenlake.zip` | Contextual DDQN trained on `Uniform[0.5, 1.0]` slip. The shocked slip 0.55 sits at the lower edge of training, so the prior is good-but-imperfect. |
    | **MCTS budget** | `m=15, d=20, c=1.4, γ=0.95` (same for MCTS-alone *and* every PA-MCTS) | Tight enough that MCTS' Q-values carry real noise -- exactly when a calibrated prior earns its keep. |

    ### What each method is doing

    - **MCTS** plans against the live post-shock snapshot every step but
      ### starts from scratch -- no prior over which actions are likely good. At m=15 the search has noticeable variance.
    - **DQN (SB3)** runs its trained policy unchanged. The new slip
      ### sits at the edge of its training distribution: routing intuition transfers, but precise tie-breaking doesn't.
    - **PA-MCTS** combines them at every step:
      ### $\;Q_\text{PA} = \alpha\, Q_\text{DDQN} + (1-\alpha)\, Q_\text{MCTS}$. The DDQN prior breaks ties when MCTS Q-values are noisy; the MCTS lookahead corrects the prior when the world has shifted under it. We sweep α ∈ {0.25, 0.50, 0.75} so you can see the trade-off.

    ### The PA-MCTS interaction loop

    ### `ns_gym.benchmark_algorithms.PAMCTS` is the canonical class. It rebuilds an MCTS tree internally each step, queries a learned Q-head, normalizes both to `[0, 1]`, and returns `α · Q_DDQN + (1−α) · Q_MCTS`. The Q-head can be integrated in two ways:

    ### **Recommended: SB3 + `StableBaselineWrapper`.** Pass any callable that returns per-action Q-values for `(state, env)` via the new `q_value_fn` kwarg. `ns_gym.base.StableBaselineWrapper` exposes `.q_values(state, env)` that does the SB3 forward pass for you; its `obs_fn=` argument handles the contextual one-hot ⊕ slip observation our DDQN was trained on:

    ```python
    import numpy as np
    from stable_baselines3 import DQN
    from ns_gym.base import StableBaselineWrapper
    from ns_gym.benchmark_algorithms import PAMCTS

    sb3 = DQN.load("model_weights/contextual_ddqn_frozenlake.zip")

    def obs_fn(state, env):                       # 19-d obs the DDQN expects
        onehot = np.eye(16, dtype=np.float32)[int(state)]
        slip = np.asarray(env.transition_prob, dtype=np.float32)
        return np.concatenate([onehot, slip])

    wrap = StableBaselineWrapper(sb3, obs_fn=obs_fn)

    agent = PAMCTS(
        alpha=0.75,                   # mixing weight on the DDQN prior
        mcts_iter=15, mcts_search_depth=20,
        mcts_discount_factor=0.95, mcts_exploration_constant=1.4,
        state_space_size=16, action_space_size=4,
        q_value_fn=wrap.q_values,     # any callable f(state, env) -> Q
    )

    obs, _ = env.reset(seed=0)
    state = obs["state"]
    done = trunc = False
    while not (done or trunc):
        action = agent.act(state, env.get_planning_env())
        obs, reward, done, trunc, _ = env.step(int(action))
        state = obs["state"]
    ```


    > **The α sweep below uses the recommended path** -- contextual
    > SB3 weights, `obs_fn` builds the 19-d input, and `PAMCTS`
    > handles the rest.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    fl_pamcts_run = mo.ui.run_button(
        label="▶ Run scenario A (PA-MCTS shock)"
    )
    fl_pamcts_run
    return (fl_pamcts_run,)


@app.cell
def _(
    DiscreteScheduler,
    DistributionStepWiseUpdate,
    NSFrozenLakeWrapper,
    build_full_pickers,
    build_ns_tensors,
    fl_pamcts_run,
    fl_planners_kwargs,
    fl_stationary_policy,
    gym,
    mo,
    ns_oracle_vi,
    parse_scenario_kwargs,
    plot_scenario_panel,
    run_scenario_metrics,
    try_load_sb3,
    type_mismatch_checker,
):
    import time as _time
    if not fl_pamcts_run.value:
        _out = mo.md("_Click run to roll **all five planners** on the shock env._")
    else:
        _t0 = _time.time()
        from ns_gym.benchmark_algorithms import MCTS as _MCTS
        try:
            from ns_gym.benchmark_algorithms import RATS as _RATS
        except Exception:
            _RATS = None

        _kw = parse_scenario_kwargs(fl_planners_kwargs.value)
        # Scenario A overrides (tuned via N=40 sweep so PA-MCTS reliably
        # beats both MCTS-alone and DQN-alone):
        #   * Use the *contextual* DDQN trained on Uniform[0.5, 1.0].
        #     Post-shock slip 0.55 sits just inside the lower edge of
        #     its training distribution, so the DDQN's prior is
        #     calibrated-but-imperfect -- exactly the regime where
        #     MCTS' online correction earns its keep.
        #   * MCTS budget m=15 so MCTS alone has enough Q-value noise
        #     for the DDQN prior to actually change the argmax.
        _kw["SB3_DQN_PATH"] = "model_weights/contextual_ddqn_frozenlake.zip"
        _kw["MCTS_KWARGS"] = dict(d=20, m=15, c=1.4, gamma=0.95)
        _dqn_model = try_load_sb3(_kw["SB3_DQN_PATH"])

        # Single-shock env: starts BELOW the contextual DDQN's training
        # range at p=0.45 (1 step OOD-low), then at step 1 a
        # DistributionStepWiseUpdate jumps P[intended] *up* to 0.55 --
        # just inside the DDQN's training band [0.5, 1.0]. The DDQN's
        # post-shock Q-values are partially-calibrated rather than
        # garbage. MCTS handles the unfamiliar geometry, DDQN supplies
        # the action prior, and PA-MCTS reliably tops both.
        def _make_env():
            base = gym.make(
                "FrozenLake-v1", is_slippery=False, max_episode_steps=50,
            )
            sched = DiscreteScheduler(event_list={1})
            update_fn = DistributionStepWiseUpdate(
                scheduler=sched, update_values=[[0.55, 0.225, 0.225]],
            )
            return NSFrozenLakeWrapper(
                base, {"P": update_fn},
                change_notification=True,
                delta_change_notification=True,
                initial_prob_dist=[0.45, 0.275, 0.275],
            )

        _MAX = 30
        _N = int(_kw["N_SEEDS"])
        _gamma = float(_kw["ORACLE_GAMMA"])

        # Build oracle V tensor for THIS scenario's env.
        _P, _R = build_ns_tensors(_make_env, _MAX, seed=0)
        _V, _ = ns_oracle_vi(_P, _R, _gamma)

        # Scenario A is the "PA-MCTS regime" -- the headline comparison
        # is MCTS vs. DQN vs. PA-MCTS at multiple mixing weights (α).
        # All four tree-searchers (MCTS + 3x PA-MCTS) use the SAME
        # MCTS_KWARGS so the only thing that varies is α. stale-VI and
        # RATS are dropped here to keep the bar chart focused on
        # whether PA-MCTS' DDQN-prior + MCTS-lookahead beats either
        # baseline alone.
        _base_planners = build_full_pickers(
            _kw, _dqn_model, fl_stationary_policy, _MCTS, _RATS,
        )
        _make_pamcts = _base_planners.pop("_make_pamcts_pick", None)
        _planners = {
            "MCTS":      _base_planners["MCTS"],
            "DQN (SB3)": _base_planners["DQN (SB3)"],
        }
        if _make_pamcts is not None:
            for _a in (0.25, 0.50, 0.75):
                _planners[f"PA-MCTS α={_a:.2f}"] = _make_pamcts(_a)

        def _progress(it):
            return mo.status.progress_bar(
                it, title="Scenario A — PA-MCTS α sweep",
                subtitle="rolling out each planner…",
                remove_on_exit=True,
            )

        _metrics, _errors = run_scenario_metrics(
            _make_env, _planners, _P, _R, _V, _gamma, _N, _MAX,
            type_mismatch_checker, progress_wrap=_progress,
        )
        _fig = plot_scenario_panel(
            _metrics, _errors,
            "Scenario A · init=[0.45, 0.275, 0.275] -> shock at t=1 to "
            "[0.55, 0.225, 0.225]  ·  contextual DDQN  ·  "
            "MCTS m=15 (same for all four)",
        )
        _err_md = (
            "\n\n_Skipped:_  " + "  ·  ".join(
                f"**{n}** ({why})" for n, why in _errors.items()
            ) if _errors else ""
        )
        _done_md = mo.md(
            f"✓ Scenario A finished in **{_time.time() - _t0:.1f} s** — "
            f"{len(_metrics)} planner(s) × {_N} seeds × {_MAX} steps."
            + _err_md
        )
        _out = mo.vstack([_fig, _done_md])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">B · RATS — bounded-but-sharp Lipschitz drift</div></div>

    ### The setup

    | Component | Choice | Why |
    | :-- | :-- | :-- |
    | **Env** | `FrozenLake-v1` (4×4, `is_slippery=False`) | Reproduces the discrete grid the RATS paper uses. |
    | **Drift** | `LCBoundedDistrubutionUpdate(L=0.05)` | Lipschitz-continuous slip drift from Lecarpentier & Rachelson 2019 — the regime RATS is designed for. |
    | **Initial slip** | `[0.7, 0.15, 0.15]` | Start in the stochastic regime (not deterministic) so the drift bites from step 0. |
    | **Rewards** | `H = -1`, `G = +1`, `F = 0`, `S = 0` | Gives RATS' shallow tree a usable gradient — without a hole penalty, every depth-2 leaf returns 0 and the worst-case minimax has nothing to discriminate. |
    | **Episode horizon** | `_MAX = 50` steps | RATS plans cautiously and takes detours; cutting at 30 amputates ~half its goal-reaching trajectories. |
    | **Agent budgets** | RATS depth=2, $L_p$=1.0  ·  MCTS d=20, m=30 | The configuration where RATS dominates the LC-drift benchmark. |

    ### What's happening at every step

    1. **Env drifts.** A fresh slip distribution is sampled from
       ### `RandomCategorical`, *but only accepted* if

       $$W_1\bigl(P_t(\cdot\mid s,a),\, P_{t+1}(\cdot\mid s,a)\bigr) \le L,$$

       ### so the slip wanders randomly but each move is bounded in 1-Wasserstein distance — no jumps larger than `L`.

    2. **Each planner sees the same snapshot.**
       ### `env.get_planning_env()` returns a deepcopy of the *current* MDP. Nobody peeks at future drift.

    3. **They differ in what they plan against:**

       - **MCTS** plans against the snapshot as if it were the future
         ### too. When reality is about to drift sharply within the ball, MCTS still picks aggressive paths because *right now* the env still looks safe.
       - **RATS** plans against the *worst* model inside a Lipschitz
         ### ball of radius $d \cdot L_p \cdot \tau$ around the snapshot (one ball per chance node, growing with depth). It pre-empts the adversarial drifts the bound permits.


    ### The RATS interaction loop

    ### RATS lives at `ns_gym.benchmark_algorithms.RATS`. Its only inputs are the live planning env and the agent's Lipschitz constants:

    ```python
    from ns_gym.benchmark_algorithms import RATS

    obs, _ = env.reset(seed=0)
    state = obs["state"]
    done = trunc = False
    while not (done or trunc):
        # Snapshot the current MDP (post-drift, post-update_fn fire)
        penv = env.get_planning_env()

        # Build a fresh RATS planner on that snapshot. L_p is the
        # *agent's* Lipschitz bound; max_depth controls tree depth.
        agent = RATS(
            action_space=penv.action_space,
            gamma=0.95, max_depth=2,
            L_p=1.0, L_r=0.0, tau=1.0,
        )
        action = agent.act(observation=state, env=penv)

        obs, reward, done, trunc, _ = env.step(int(action))
        state = obs["state"]
    ```

    > **Note.** RATS reads rewards from `penv.unwrapped.P[s][a][i][2]`,
    > i.e. the env's transition table. On sparse-reward FrozenLake the
    > tutorial scenarios inject a hole=-1 penalty into the planning
    > env's `P` so the depth-2 worst-case minimax has a usable
    > gradient. See `_inject_hole_penalty` in the scenario cell.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    fl_rats_run = mo.ui.run_button(label="▶ Run scenario B (RATS Lipschitz)")
    fl_rats_run
    return (fl_rats_run,)


@app.cell
def _(
    ContinuousScheduler,
    LCBoundedDistrubutionUpdate,
    NSFrozenLakeWrapper,
    build_full_pickers,
    build_ns_tensors,
    fl_planners_kwargs,
    fl_rats_run,
    fl_stationary_policy,
    gym,
    mo,
    ns_oracle_vi,
    parse_scenario_kwargs,
    plot_scenario_panel,
    run_scenario_metrics,
    try_load_sb3,
    type_mismatch_checker,
):
    import time as _time
    if not fl_rats_run.value:
        _out = mo.md(
            "_Click run to roll **all five planners** on a Lipschitz-bounded "
            "random drift (Lecarpentier & Rachelson 2019)._"
        )
    else:
        _t0 = _time.time()
        from ns_gym.benchmark_algorithms import MCTS as _MCTS
        try:
            from ns_gym.benchmark_algorithms import RATS as _RATS
        except Exception:
            _RATS = None

        _kw = parse_scenario_kwargs(fl_planners_kwargs.value)
        _dqn_model = try_load_sb3(_kw["SB3_DQN_PATH"])
        _ENV_L = float(_kw.get("ENV_L", 0.15))

        # The "real" env everyone is *scored* on. Sparse rewards
        # (goal=+1, hole=0, frozen=0) so all five planners' cum_reward
        # lives in [0, 1] -- a fair, comparable y-axis.
        # initial_prob_dist=[0.7, 0.15, 0.15] starts in the stochastic
        # regime so the LC drift bites immediately.
        def _make_env():
            base = gym.make(
                "FrozenLake-v1", is_slippery=False, max_episode_steps=50,
            )
            update_fn = LCBoundedDistrubutionUpdate(
                scheduler=ContinuousScheduler(), L=_ENV_L,
            )
            return NSFrozenLakeWrapper(
                base, {"P": update_fn},
                change_notification=True,
                delta_change_notification=True,
                initial_prob_dist=[0.7, 0.15, 0.15],
            )

        # RATS plans against a *private* hole-penalty view of the same
        # snapshot. Without a negative hole signal, every depth-2 leaf
        # of RATS' tree returns 0 (goal is too far to see) and the
        # worst-case minimax has nothing to discriminate. We mutate the
        # planning env's P to inject hole=-1, goal=+1; the real env it
        # acts in still emits sparse rewards, so RATS is scored on the
        # same scale as MCTS / PA-MCTS / DQN.
        def _inject_hole_penalty(penv):
            # Walk the transition table once and replace the reward field
            # of every absorbing-with-zero-reward transition (the hole
            # absorbing self-loops + transitions INTO holes) with -1.
            # Goal transitions (done=True with r>0) are left untouched.
            P = penv.unwrapped.P
            new_P = {}
            for s in P:
                new_P[s] = {}
                for a in P[s]:
                    rebuilt = []
                    for prob, ns, r, done in P[s][a]:
                        if done and r <= 0:
                            rebuilt.append((prob, ns, -1.0, done))
                        else:
                            rebuilt.append((prob, ns, float(r), done))
                    new_P[s][a] = rebuilt
            # IMPORTANT: NSFrozenLakeWrapper.__deepcopy__ (called by
            # RATS.act -> deepcopy(env)) rebuilds unwrapped.P from
            # `self.P` (the wrapper attribute). So we must update BOTH
            # the wrapper's .P and the unwrapped env's .P, otherwise the
            # mutation is silently lost and RATS plans against sparse
            # rewards (= no gradient = picks arbitrarily = 0% goals).
            penv.P = new_P
            penv.unwrapped.P = new_P
            penv.intial_p = new_P
            return penv

        def _rats_pick(state, _t, env):
            if _RATS is None:
                raise RuntimeError("RATS unavailable")
            penv = _inject_hole_penalty(env.get_planning_env())
            return int(_RATS(
                action_space=penv.action_space, **_kw["RATS_KWARGS"],
            ).act(observation=int(state), env=penv))

        # RATS plans cautiously and detours -- shorter horizons amputate
        # ~half its goal-reaching trajectories, so give it 50 steps.
        _MAX = 50
        _N = int(_kw["N_SEEDS"])
        _gamma = float(_kw["ORACLE_GAMMA"])

        _P, _R = build_ns_tensors(_make_env, _MAX, seed=0)
        _V, _ = ns_oracle_vi(_P, _R, _gamma)

        _planners = build_full_pickers(
            _kw, _dqn_model, fl_stationary_policy, _MCTS, _RATS,
        )
        # build_full_pickers exposes a "_make_pamcts_pick" factory used by
        # scenario A's α sweep -- not a real picker, so drop it before
        # passing the dict to run_scenario_metrics.
        _planners.pop("_make_pamcts_pick", None)
        # Override RATS' picker with the hole-penalty-augmented version.
        if "RATS" in _planners:
            _planners["RATS"] = _rats_pick

        def _progress(it):
            return mo.status.progress_bar(
                it, title="Scenario B — RATS Lipschitz",
                subtitle="rolling out each planner…",
                remove_on_exit=True,
            )

        _metrics, _errors = run_scenario_metrics(
            _make_env, _planners, _P, _R, _V, _gamma, _N, _MAX,
            type_mismatch_checker, progress_wrap=_progress,
        )
        _fig = plot_scenario_panel(
            _metrics, _errors,
            f"Scenario B · Lipschitz-bounded random drift (env L={_ENV_L:.2f})",
        )
        _err_md = (
            "\n\n_Skipped:_  " + "  ·  ".join(
                f"**{n}** ({why})" for n, why in _errors.items()
            ) if _errors else ""
        )
        _done_md = mo.md(
            f"✓ Scenario B finished in **{_time.time() - _t0:.1f} s** — "
            f"{len(_metrics)} planner(s) × {_N} seeds × {_MAX} steps."
            + _err_md
        )
        _out = mo.vstack([_fig, _done_md])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 1.6em 0 0.5em 0; padding: 0.4em 0.8em; border-left: 4px solid #5a3a8a; background: #f6f2fb;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">C · Contextual MDPs — known new context per episode, stationary within</div></div>

    ### A different flavor of non-stationarity. Each episode draws a new MDP $M_c$ from a *known* family $\{M_c\}_{c \in \mathcal{C}}$ — here indexed by the slip parameter `p_intended`. **Within** an episode the dynamics are stationary; **across** episodes the context $c$ changes and the agent is told the new $c$ at episode start. Who's in the bake-off

    | Family | Members | Knows c? | Adapts at decision time? |
    | :-- | :-- | :-- | :-- |
    | **Oracle VI** | one VI solve per context | yes (rebuilds Q\* per c) | n/a |
    | **MCTS** | UCB1 tree from current snapshot | implicit (via `env.P`) | yes |
    | **RATS** | min-max tree, $L_p$=1 | implicit (via `env.P`) | yes |
    | **PA-MCTS** | mix of contextual DDQN prior + MCTS Q | yes (DDQN reads $c$) | yes |
    | **DQN ctx** | contextual DDQN, trained on $p \sim \text{Uniform}[0.5, 1.0]$ | yes (input includes $c$) | no (one forward pass) |
    | **DQN p=0.70** | stationary DDQN trained at $p$=0.7 only | no (ignores $c$) | no |
    | **DQN p=0.33** | stationary DDQN trained at $p$≈0.33 only | no (ignores $c$) | no |

    ### The two stationary DDQNs are baselines: they show what happenswhen you train at *one* context and deploy across a distribution.

    ### What the plot shows

    ### For each context $c \in \{0.4, 0.55, 0.7, 0.85, 0.95\}$ we run `N_SEEDS` episodes per planner and report the mean cumulative reward. Each line is one planner's curve over contexts; bands are standard errors. Oracle VI sets the per-context ceiling.

    ### The DQN interaction loop

    ### The DDQN families are even simpler — load the SB3 weights, build the same observation vector the network was trained on (one-hot state ⊕ slip context), and ask for the deterministic action:

    ```python
    import numpy as np
    from stable_baselines3 import DQN

    # Pick whichever you want to demo: contextual or stationary
    dqn = DQN.load("model_weights/contextual_ddqn_frozenlake.zip")
    # dqn = DQN.load("model_weights/ddqn_stationary_p0.700.zip")
    # dqn = DQN.load("model_weights/ddqn_stationary_p0.333.zip")

    def sb3_obs(state, env):
        onehot = np.eye(16, dtype=np.float32)[int(state)]
        slip = np.asarray(env.transition_prob, dtype=np.float32)
        return np.concatenate([onehot, slip])

    obs, _ = env.reset(seed=0)
    state = obs["state"]
    done = trunc = False
    while not (done or trunc):
        action, _ = dqn.predict(sb3_obs(state, env), deterministic=True)
        obs, reward, done, trunc, _ = env.step(int(action))
        state = obs["state"]
    ```

    ### The contextual DDQN reads `env.transition_prob` directly so its policy adapts when you change the context between episodes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    fl_context_run = mo.ui.run_button(
        label="▶ Run scenario C (contextual MDPs)"
    )
    fl_context_run
    return (fl_context_run,)


@app.cell
def _(
    ContinuousScheduler,
    DistributionNoUpdate,
    NSFrozenLakeWrapper,
    build_full_pickers,
    build_ns_tensors,
    fl_context_run,
    fl_planners_kwargs,
    fl_stationary_policy,
    gym,
    mo,
    np,
    ns_oracle_vi,
    parse_scenario_kwargs,
    plt,
    try_load_sb3,
    type_mismatch_checker,
):
    import time as _time
    if not fl_context_run.value:
        _out = mo.md(
            "_Click run to evaluate every planner across a 5-point context "
            "grid (slip ∈ {0.40, 0.55, 0.70, 0.85, 0.95})._"
        )
    else:
        _t0 = _time.time()
        import torch
        from ns_gym.benchmark_algorithms import MCTS as _MCTS
        try:
            from ns_gym.benchmark_algorithms import RATS as _RATS
        except Exception:
            _RATS = None

        _kw = parse_scenario_kwargs(fl_planners_kwargs.value)
        # Scenario C overrides:
        #   * Bump MCTS budget so MCTS-alone is competitive on the
        #     contextual baseline (scenario A intentionally throttles
        #     MCTS to expose PA-MCTS' advantage; scenario C's question
        #     is the opposite -- "how do these methods compare when each
        #     gets a fair search budget").
        _kw["MCTS_KWARGS"] = dict(d=20, m=30, c=1.4, gamma=0.95)
        _N = int(_kw["N_SEEDS"])
        _MAX = 30
        _gamma = float(_kw["ORACLE_GAMMA"])

        # ---- 5-point context grid ----
        _CONTEXTS = [0.40, 0.55, 0.70, 0.85, 0.95]

        # ---- RATS gets a private hole-penalty view of the planning env
        # (same trick as scenario B). Without this, every RATS leaf at
        # depth 2 returns 0 reward (goal too far) and the worst-case
        # minimax has nothing to discriminate -- RATS picks arbitrary
        # actions and gets 0 across all contexts, defeating the point
        # of the comparison.
        def _inject_hole_penalty(penv):
            P = penv.unwrapped.P
            new_P = {
                s: {
                    a: [
                        (p, ns, -1.0 if (d and r <= 0) else float(r), d)
                        for p, ns, r, d in P[s][a]
                    ] for a in P[s]
                } for s in P
            }
            penv.P = new_P
            penv.unwrapped.P = new_P
            penv.intial_p = new_P
            return penv

        def _make_rats_picker():
            if _RATS is None:
                return None
            def _pick(state, _t, env):
                penv = _inject_hole_penalty(env.get_planning_env())
                return int(_RATS(
                    action_space=penv.action_space,
                    **_kw["RATS_KWARGS"],
                ).act(observation=int(state), env=penv))
            return _pick

        # ---- load all three DDQNs once ----
        _dqn_ctx = try_load_sb3("model_weights/contextual_ddqn_frozenlake.zip")
        _dqn_07  = try_load_sb3("model_weights/ddqn_stationary_p0.700.zip")
        _dqn_03  = try_load_sb3("model_weights/ddqn_stationary_p0.333.zip")

        # ---- per-context stationary env factory ----
        def _make_env_for(p):
            base = gym.make(
                "FrozenLake-v1", is_slippery=False, max_episode_steps=50,
            )
            sched = ContinuousScheduler()
            upd = DistributionNoUpdate(scheduler=sched)
            return NSFrozenLakeWrapper(
                base, {"P": upd},
                change_notification=False,
                delta_change_notification=False,
                initial_prob_dist=[p, (1 - p) / 2, (1 - p) / 2],
            )

        # ---- per-DDQN picker factory (uses live env's slip as context) ----
        def _sb3_obs(state, env):
            onehot = np.eye(16, dtype=np.float32)[int(state)]
            pvec = np.asarray(env.transition_prob, dtype=np.float32)
            if pvec.size != 3:
                pvec = np.array(
                    [pvec[0], (1 - pvec[0]) / 2, (1 - pvec[0]) / 2],
                    dtype=np.float32,
                )
            return np.concatenate([onehot, pvec])

        def _make_dqn_picker(dqn):
            if dqn is None:
                return None
            def _pick(state, _t, env):
                a, _ = dqn.predict(_sb3_obs(state, env), deterministic=True)
                return int(a)
            return _pick

        # ---- per-context VI oracle (env is stationary, so one Q tensor
        # suffices per context; no per-seed rebuild needed) ----
        def _build_oracle(env_factory):
            P, R = build_ns_tensors(env_factory, _MAX, seed=0)
            V, _ = ns_oracle_vi(P, R, _gamma)
            return P, R, V

        def _make_oracle_picker(P, R, V):
            T = V.shape[0] - 1
            def _pick(state, t, _env):
                t_idx = min(t, T - 1)
                Q = np.einsum(
                    "ap,ap->a",
                    P[t_idx, int(state)],
                    R[t_idx, int(state)] + _gamma * V[t_idx + 1],
                )
                return int(np.argmax(Q))
            return _pick

        # ---- per-context evaluation ----
        def _rollout(picker, env_factory, n_seeds):
            returns = []
            for s in range(n_seeds):
                env = env_factory()
                obs, _ = env.reset(seed=s)
                state, _ = type_mismatch_checker(obs)
                np.random.seed(s)
                cum_r = 0.0
                done = trunc = False
                t = 0
                while not (done or trunc) and t < _MAX:
                    a = int(picker(int(state), t, env))
                    obs, reward, done, trunc, _ = env.step(a)
                    state, reward = type_mismatch_checker(obs, reward)
                    cum_r += float(reward)
                    t += 1
                returns.append(cum_r)
            return np.asarray(returns)

        # Planner names + colors (extends the global palette).
        _planner_names = [
            "Oracle VI", "MCTS", "RATS",
            "PA-MCTS", "DQN ctx", "DQN p=0.70", "DQN p=0.33",
        ]
        _palette = {
            "Oracle VI":   "#1a1a1a",
            "MCTS":        "tab:blue",
            "RATS":        "tab:purple",
            "PA-MCTS":     "tab:orange",
            "DQN ctx":     "tab:green",
            "DQN p=0.70":  "#7fbf7b",   # lighter green
            "DQN p=0.33":  "#1b9e77",   # darker green
        }

        # results[name] = list of (mean_return, sem) per context
        _results = {n: [] for n in _planner_names}
        _errors = {}

        _it = mo.status.progress_bar(
            _CONTEXTS,
            title="Scenario C — contextual MDPs",
            subtitle="rolling out each context…",
            remove_on_exit=True,
        )
        for _p in _it:
            _factory = lambda p=_p: _make_env_for(p)

            # Build per-context oracle.
            _P, _R, _V = _build_oracle(_factory)

            # Assemble pickers for THIS context.
            _base = build_full_pickers(
                _kw, _dqn_ctx, fl_stationary_policy, _MCTS, _RATS,
            )
            _base.pop("_make_pamcts_pick", None)
            _picks = {
                "Oracle VI":  _make_oracle_picker(_P, _R, _V),
                "MCTS":       _base["MCTS"],
                "RATS":       _make_rats_picker(),  # hole-penalty version
                "PA-MCTS":    _base["PA-MCTS"],   # uses contextual DDQN
                "DQN ctx":    _make_dqn_picker(_dqn_ctx),
                "DQN p=0.70": _make_dqn_picker(_dqn_07),
                "DQN p=0.33": _make_dqn_picker(_dqn_03),
            }

            for _name in _planner_names:
                _pk = _picks.get(_name)
                if _pk is None:
                    _errors[_name] = "unavailable (model not loaded)"
                    _results[_name].append((np.nan, np.nan))
                    continue
                try:
                    _rs = _rollout(_pk, _factory, _N)
                    _mu = float(np.mean(_rs))
                    _se = float(np.std(_rs, ddof=1) / np.sqrt(len(_rs))) if len(_rs) > 1 else 0.0
                    _results[_name].append((_mu, _se))
                except Exception as _e:
                    _errors[_name] = f"{type(_e).__name__}: {_e}"
                    _results[_name].append((np.nan, np.nan))

        # ---- plot: lines per planner, x = context, y = mean reward ----
        _fig, _ax = plt.subplots(figsize=(11, 4.6), layout="constrained")
        _xs = np.asarray(_CONTEXTS)
        for _name in _planner_names:
            _vals = _results[_name]
            if not _vals:
                continue
            _mus = np.asarray([v[0] for v in _vals])
            _ses = np.asarray([v[1] for v in _vals])
            if np.isnan(_mus).all():
                continue
            _color = _palette.get(_name, "gray")
            _ls = "-" if not _name.startswith("DQN p") else "--"
            # Shaded ±1 SE band behind the mean line -- visually quieter
            # than errorbar caps and plays nicer when several lines
            # overlap. Mean line stays on top via zorder.
            _ax.fill_between(
                _xs, _mus - _ses, _mus + _ses,
                color=_color, alpha=0.15, linewidth=0, zorder=1,
            )
            _ax.plot(
                _xs, _mus, label=_name, color=_color,
                linestyle=_ls, linewidth=1.8, marker="o", markersize=5,
                zorder=3,
            )

        _ax.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
        _ax.set_xlabel("context  c  =  P[intended]  (slip parameter)")
        _ax.set_ylabel(f"mean cumulative reward (over {_N} seeds)")
        _ax.set_xticks(_xs)
        _ax.set_xticklabels([f"{x:.2f}" for x in _xs])
        _ax.set_title(
            "Scenario C · Contextual MDPs  ·  performance across slip contexts  ·  "
            f"each context: stationary, N={_N} seeds, {_MAX} steps",
            fontsize=10, fontweight=600,
        )
        _ax.set_ylim(-0.05, 1.05)
        _ax.grid(alpha=0.25)
        _ax.legend(fontsize=8, loc="lower right", framealpha=0.92)

        _err_md = ""
        if _errors:
            _err_md = "\n\n_Skipped:_  " + "  ·  ".join(
                f"**{n}** ({why})" for n, why in _errors.items()
            )
        _done_md = mo.md(
            f"✓ Scenario C finished in **{_time.time() - _t0:.1f} s** — "
            f"{len(_planner_names)} planners × {len(_CONTEXTS)} contexts × "
            f"{_N} seeds × {_MAX} steps." + _err_md
        )
        _out = mo.vstack([_fig, _done_md])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <div style="
        margin: 5em 0 1.5em 0;
        padding: 2.5em 2em;
        border-top: 8px solid #1a1a1a;
        border-bottom: 8px solid #1a1a1a;
        background: linear-gradient(180deg, #f5f5f5, #fafafa);
    ">
      <div style="font-size: 0.9em; opacity: 0.55;
                  letter-spacing: 0.2em; font-weight: 600;
                  color: #1a1a1a;">
        CAPSTONE &middot; ACTIVITY
      </div>
      <div style="font-size: 2.7em; font-weight: 800;
                  margin-top: 0.25em; line-height: 1.05;">
        Put it all together — your env, your policy, your metric
      </div>
      <div style="font-size: 1.05em; opacity: 0.72;
                  margin-top: 0.7em; max-width: 56em;">
        One open-ended exercise that ties every module together.
        Build a non-stationary env, declare your modeling assumptions,
        pick a policy, and write the metric that scores it.
      </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="margin: 2.2em 0 0.8em 0; padding: 0.6em 1em 0.7em 1em; border-left: 5px solid #1a1a1a; background: #efefef;"><div style="font-size: 2.5em; font-weight: 700; line-height: 1.2;">🎯 Your turn — env + assumptions + policy + metric, end-to-end</div></div>

    ### The four reference tabs map onto the four steps you've seen throughout the session — the **✏️ Your code** tab is where you combine them. Aim for an end-to-end script that:

    1. **Creates an NS env.** Pick a base Gymnasium env, declare
       ### *what* drifts (the tunable parameter), wire a scheduler (*when*) and an update-function (*how*), then wrap with the appropriate `NS*Wrapper`.
    2. **States its assumptions.** What `change_notification` /
       ### `delta_change_notification` level does the agent get? What does `env.get_planning_env()` return? *Comment these next to the wrapper call so a reader doesn't have to guess.*
    3. **Picks a policy.** Anything from
       ### `ns_gym.benchmark_algorithms` (`MCTS`, `PAMCTS`, `RATS`, `DQNAgent`) or your own — random baseline, hand-coded heuristic, SB3 model loaded with `DQN.load(...)`. The interaction loop is always the same shape: reset → loop step → close.
    4. **Computes a performance metric.** Don't just print
       ### `ep_return`. Roll out *N* seeds, capture per-step diagnostics, and report at least one of: mean-return ± std, cumulative Q-gap (if you have an oracle), success rate, time-to-first-goal, parameter trajectory.

    ### The reference tabs cover three canonical recipes — CartPole + MCTS (the canonical onboarding example), Bridge + MCTS (a tabular env where you *can* compute Q-gap), and pretrained SB3 DQN. Use them as scaffolding and edit the **✏️ Your code** tab freely.

    > **Rubric — what a complete answer looks like.** The starter runs
    > out-of-the-box; that's not the deliverable. Aim for a script
    > that hits *all four* of these:
    >
    > **R1 · Picks a non-trivial NS-MDP.** The drift should actually
    > break a stationary policy. If you keep the default
    > `IncrementUpdate(k=0.05)` the env barely moves. Bump the rate,
    > or pick `SigmoidTransition` / `RandomWalk` so a stale policy
    > visibly degrades.
    >
    > **R2 · Justifies the notification level in a comment.** Why
    > `change_notification=True` and not `False`? Why
    > `delta_change_notification=False` (or `True`)? Two lines max,
    > but the reader should know which level you chose and why.
    >
    > **R3 · Uses a *real* policy.** Random baseline is fine *for
    > comparison*, but the headline policy should be MCTS / PA-MCTS /
    > RATS / pretrained SB3 — something whose decision actually
    > depends on the model.
    >
    > **R4 · Reports more than one number.** At minimum: mean ± std
    > over N seeds. Better: also one of (success rate, time-to-first-
    > goal, cumulative parameter trajectory, Q-gap if you have an
    > oracle). A single `print(ep_return)` doesn't pass.
    >
    > Bonus: run *two* policies and print which one wins on each
    > metric — that's exactly the comparison shape Module 4 builds.
    """)
    return


@app.cell(hide_code=True)
def _(fl_scheduler_editor, fl_update_editor, mo):
    from textwrap import dedent as _dedent

    _starter = _dedent('''\
        # CAPSTONE — combine env + assumptions + policy + metric.
        # Sandbox imports already wired: gym, ns_gym, np
        #
        # 1) BUILD THE ENV   — pick base env, scheduler, update fn, wrapper
        # 2) STATE ASSUMPTIONS — comment what the agent sees (notification level,
        #                        planning env, parameter access)
        # 3) PICK A POLICY   — MCTS / PAMCTS / RATS / random / SB3 DQN / your own
        # 4) COMPUTE METRICS — roll N seeds, report mean ± std (and more)
        import gymnasium as gym
        from ns_gym.wrappers import NSClassicControlWrapper
        from ns_gym.schedulers import ContinuousScheduler
        from ns_gym.update_functions import IncrementUpdate
        from ns_gym.benchmark_algorithms import MCTS

        # --- 1) ENV -----------------------------------------------------------
        sched = ContinuousScheduler()
        update_fn = IncrementUpdate(sched, k=0.05)   # gravity drifts each step

        # --- 2) ASSUMPTIONS ---------------------------------------------------
        # - Single tunable parameter: gravity
        # - Notification level: change + delta_change → agent SEES gravity moved
        # - get_planning_env() returns the env with the *current* gravity baked in
        def make_env():
            base = gym.make("CartPole-v1")
            return NSClassicControlWrapper(
                base, {"gravity": update_fn},
                change_notification=True,
                delta_change_notification=True,
            )

        # --- 3) POLICY (rebuilt every step so it sees the live model) ---------
        def pick_action(state, env_):
            penv = env_.get_planning_env()
            agent = MCTS(penv, state=state, d=20, m=20, c=1.4, gamma=0.99)
            a, _ = agent.search()
            return int(a)

        # --- 4) METRIC: mean ± std over N seeds -------------------------------
        import numpy as np
        N_SEEDS, MAX_STEPS = 5, 50
        returns = []
        for s in range(N_SEEDS):
            e = make_env()
            obs, _ = e.reset(seed=s)
            state = obs["state"]
            ep_r, t = 0.0, 0
            done = trunc = False
            while not (done or trunc) and t < MAX_STEPS:
                a = pick_action(state, e)
                obs, r, done, trunc, _ = e.step(a)
                state = obs["state"]
                ep_r += float(r)
                t += 1
            returns.append(ep_r)

        mean, std = float(np.mean(returns)), float(np.std(returns))
        print(f"return mean ± std over {N_SEEDS} seeds : {mean:.2f} ± {std:.2f}")
        print(f"raw returns                            : {returns}")
        ''')

    _github = _dedent('''\
        # Reference 1 — canonical NS-Gym onboarding example.
        # Source: https://github.com/scope-lab-vu/ns_gym
        import gymnasium as gym
        from ns_gym.wrappers import NSClassicControlWrapper
        from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
        from ns_gym.update_functions import RandomWalk, IncrementUpdate
        from ns_gym.benchmark_algorithms import MCTS

        env = gym.make("CartPole-v1")
        sched_1 = ContinuousScheduler()
        sched_2 = PeriodicScheduler(period=3)
        upd_1 = IncrementUpdate(sched_1, k=0.1)
        upd_2 = RandomWalk(sched_2)
        tunable_params = {"masspole": upd_1, "gravity": upd_2}
        ns_env = NSClassicControlWrapper(
            env, tunable_params, change_notification=True,
        )

        obs, _ = ns_env.reset()
        planning_env = ns_env.get_planning_env()
        agent = MCTS(
            planning_env, state=obs["state"],
            d=50, m=100, c=1.4, gamma=0.99,
        )

        ep_return, t = 0.0, 0
        done = trunc = False
        while not (done or trunc) and t < 50:
            action, _ = agent.search()
            obs, reward, done, trunc, _ = ns_env.step(action)
            ep_return += float(reward)
            planning_env = ns_env.get_planning_env()
            t += 1

        print(f"Episode Reward: {ep_return}  (took {t} steps)")
        ''')

    _bridge = _dedent('''\
        # Reference 2 — Bridge + MCTS (tabular, has an oracle if you compute one).
        import gymnasium as gym
        from ns_gym.wrappers import NSBridgeWrapper
        from ns_gym.schedulers import ContinuousScheduler
        from ns_gym.update_functions import DistributionDecrementUpdate
        from ns_gym.benchmark_algorithms import MCTS

        env = gym.make("ns_gym/Bridge-v0", max_episode_steps=50, split_probs=True)
        sched_l, sched_r = ContinuousScheduler(), ContinuousScheduler()
        ns_env = NSBridgeWrapper(
            env,
            tunable_params={
                "P_left":  DistributionDecrementUpdate(sched_l, k=0.02),
                "P_right": DistributionDecrementUpdate(sched_r, k=0.20),
            },
            change_notification=True,
            delta_change_notification=True,
            initial_prob_dist=([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        )

        obs, _ = ns_env.reset(seed=0)
        planning_env = ns_env.get_planning_env()
        agent = MCTS(
            planning_env, state=obs["state"],
            d=30, m=50, c=1.4, gamma=0.95,
        )

        ep_return, t = 0.0, 0
        done = trunc = False
        while not (done or trunc):
            action, _ = agent.search()
            obs, reward, done, trunc, _ = ns_env.step(action)
            ep_return += float(reward)
            planning_env = ns_env.get_planning_env()
            agent = MCTS(
                planning_env, state=obs["state"],
                d=30, m=50, c=1.4, gamma=0.95,
            )
            t += 1

        print(f"Bridge MCTS — return={ep_return:+.1f}, steps={t}")
        ''')

    _sb3 = _dedent('''\
        # Reference 3 — pretrained Stable-Baselines3 DQN on a drifting env.
        import gymnasium as gym
        from stable_baselines3 import DQN
        from ns_gym.wrappers import NSFrozenLakeWrapper
        from ns_gym.schedulers import ContinuousScheduler
        from ns_gym.update_functions import DistributionDecrementUpdate

        model = DQN.load("model_weights/ddqn_stationary_p0.333.zip")

        env = gym.make("FrozenLake-v1", is_slippery=False, max_episode_steps=50)
        ns_env = NSFrozenLakeWrapper(
            env,
            {"P": DistributionDecrementUpdate(ContinuousScheduler(), k=0.05)},
            change_notification=True,
            delta_change_notification=True,
            initial_prob_dist=[1.0, 0.0, 0.0],
        )

        obs, _ = ns_env.reset(seed=0)
        state = obs["state"]
        ep_return, t = 0.0, 0
        done = trunc = False
        while not (done or trunc):
            action, _ = model.predict(state, deterministic=True)
            obs, reward, done, trunc, _ = ns_env.step(int(action))
            state = obs["state"]
            ep_return += float(reward)
            t += 1

        print(f"SB3 DQN on NS-FrozenLake — return={ep_return:.2f}, steps={t}")
        ''')

    # Stitch the user's saved scheduler + update classes into a full
    # end-to-end starter so the capstone editor can extend their own
    # definitions instead of copying boilerplate. We read straight from
    # the editor values (the source-of-truth strings) so anything the
    # user typed in Module 1 flows through.
    _your_saved = _dedent('''\
        # Reference — your saved FLUserScheduler + FLUserUpdate wired
        # into a full FrozenLake NS-MDP. Edit freely.
        import gymnasium as gym
        import numpy as np
        from ns_gym import base as nsg_base
        from ns_gym.wrappers import NSFrozenLakeWrapper
        from ns_gym.benchmark_algorithms import MCTS

        # ----- your saved scheduler (from Module 1) -----
        {sched_src}

        # ----- your saved update fn (from Module 1) -----
        {upd_src}

        # ----- env + agent ------------------------------------------------
        def make_env():
            base = gym.make(
                "FrozenLake-v1", is_slippery=False, max_episode_steps=50,
            )
            sched = FLUserScheduler()
            upd = FLUserUpdate(sched)
            return NSFrozenLakeWrapper(
                base, {{"P": upd}},
                change_notification=True,
                delta_change_notification=True,
                initial_prob_dist=[1.0, 0.0, 0.0],
            )

        # ----- 5-seed MCTS rollout ---------------------------------------
        N_SEEDS, MAX_STEPS = 5, 30
        returns = []
        for s in range(N_SEEDS):
            env = make_env()
            obs, _ = env.reset(seed=s)
            state = obs["state"]
            ep_r, t = 0.0, 0
            done = trunc = False
            while not (done or trunc) and t < MAX_STEPS:
                penv = env.get_planning_env()
                agent = MCTS(
                    penv, state=int(state),
                    d=20, m=40, c=1.4, gamma=0.95,
                )
                a, _ = agent.search()
                obs, r, done, trunc, _ = env.step(int(a))
                state = obs["state"]
                ep_r += float(r)
                t += 1
            returns.append(ep_r)

        print(f"returns over {{N_SEEDS}} seeds: {{returns}}")
        print(f"mean ± std: {{np.mean(returns):.2f}} ± {{np.std(returns):.2f}}")
        ''').format(
        sched_src=fl_scheduler_editor.value.rstrip(),
        upd_src=fl_update_editor.value.rstrip(),
    )

    fl_freeform_editor = mo.ui.code_editor(
        value=_starter, language="python", min_height=28,
    )
    _ref_github = mo.ui.code_editor(
        value=_github, language="python", disabled=True, min_height=22,
    )
    _ref_bridge = mo.ui.code_editor(
        value=_bridge, language="python", disabled=True, min_height=22,
    )
    _ref_sb3 = mo.ui.code_editor(
        value=_sb3, language="python", disabled=True, min_height=22,
    )
    _ref_your = mo.ui.code_editor(
        value=_your_saved, language="python", disabled=True, min_height=28,
    )
    mo.ui.tabs({
        "✏️ Your code": fl_freeform_editor,
        "🧪 Your saved scheduler/update": _ref_your,
        "📘 CartPole + MCTS (GitHub example)": _ref_github,
        "🌉 Bridge + MCTS": _ref_bridge,
        "📦 Pretrained DQN (SB3)": _ref_sb3,
    })
    return (fl_freeform_editor,)


@app.cell(hide_code=True)
def _(mo):
    fl_freeform_run = mo.ui.run_button(label="▶ Run your capstone script")
    fl_freeform_run
    return (fl_freeform_run,)


@app.cell
def _(fl_freeform_editor, fl_freeform_run, gym, mo, np):
    import time as _time
    if not fl_freeform_run.value:
        _out = mo.md("_Click the button above to execute the code in the active tab._")
    else:
        import io as _io
        import sys as _sys
        import ns_gym as _ns_gym

        _ns = {
            "__name__": "__sandbox__",
            "gym": gym,
            "ns_gym": _ns_gym,
            "np": np,
        }
        _stdout = _io.StringIO()
        _stderr = _io.StringIO()
        _so, _se = _sys.stdout, _sys.stderr
        _sys.stdout, _sys.stderr = _stdout, _stderr
        _t0 = _time.time()
        try:
            exec(fl_freeform_editor.value, _ns)
            _status = "✓ ran cleanly"
        except Exception as _e:
            _status = f"⚠ {type(_e).__name__}: {_e}"
        finally:
            _sys.stdout, _sys.stderr = _so, _se
        _elapsed = _time.time() - _t0

        _out = mo.vstack([
            mo.md(f"**{_status}** — finished in **{_elapsed:.2f} s**"),
            mo.md("```\n" + (_stdout.getvalue() or "(no stdout)") + "\n```"),
            *([mo.md(f"⚠ stderr:\n```\n{_stderr.getvalue()}\n```")]
              if _stderr.getvalue() else []),
        ])
    _out
    return


if __name__ == "__main__":
    app.run()
