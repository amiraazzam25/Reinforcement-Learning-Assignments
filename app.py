"""Interactive Streamlit dashboard running the user's real Python Q-Learning logic.
Run locally with: python -m streamlit run app.py
"""
from __future__ import annotations

from html import escape
import random

import pandas as pd
import streamlit as st

from heater_rl import (
    ACTION_NAMES, ACTIONS, EXPECTED_POLICY, HEATER_OFF, HEATER_ON, INITIAL_STATE,
    STATES, blank_q, choose_action, evaluate, greedy_action, policy_correct_count,
    q_mae, reward_function, theoretical_q, train, transition, update_q,
)

st.set_page_config(
    page_title="HeatWise | Q-Learning Control Lab", page_icon="🌡️",
    layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {--bg:#0b1425; --panel:#17253a; --teal:#69e4ce; --muted:#a1b4cd;}
.stApp {background:radial-gradient(ellipse at 70% -20%,#173955 0%,transparent 55%), #0b1425; color:#e9f2ff;}
section[data-testid="stSidebar"] {background:#0e1b2d; border-right:1px solid #2d4260;}
[data-testid="stMetric"] {background:#14253c;border:1px solid #2b435f;border-radius:16px;padding:14px;}
[data-testid="stMetricLabel"] {color:#b7cbe5 !important;}
[data-testid="stMetricValue"] {color:#effbff !important;}
[data-testid="stVerticalBlock"]>div:has(>.app-eyebrow) {padding-top:0;}
.app-eyebrow {color:#60dcc8;font-size:.8rem;font-weight:800;letter-spacing:.17rem;margin:0 0 .2rem;}
.hero-title {font-size:2.35rem;letter-spacing:-.035em;font-weight:850;line-height:1.05;margin:.12rem 0 .65rem;color:#f5fbff;}
.hero-desc {color:#b0c4df;font-size:1rem;margin-bottom:1.2rem;}
.info-band {background:#14263e;border:1px solid #2b4766;padding:12px 16px;border-radius:14px;color:#cadaee;}
.room-scene {position:relative;min-height:310px;overflow:hidden;border-radius:22px;border:1px solid #4a6077;background:linear-gradient(180deg,#1b4366 0%,#16334e 75%,#0e1b2f 75%);margin-top:7px;box-shadow:0 20px 70px #060e1e60;}
.room-scene.comfortable {background:linear-gradient(180deg,#204b4b 0%,#18363e 75%,#111e31 75%);}
.room-scene.hot {background:linear-gradient(180deg,#633b47 0%,#38283d 75%,#191b31 75%);}
.room-text {position:absolute;top:22px;left:24px;font-weight:800;font-size:1.45rem;color:#f7fbff;}
.room-sub {position:absolute;top:61px;left:24px;color:#c5d6e8;font-size:.83rem;}
.room-cloud {position:absolute;right:23px;top:25px;font-size:2.8rem;}
.window {position:absolute;left:9%;top:99px;width:130px;height:130px;border:9px solid #b6d4e065;border-radius:11px;background:linear-gradient(135deg,#0d2a4e,#366d85);box-shadow:0 0 32px #90cdf22a;}
.window:before {content:'';position:absolute;left:50%;height:100%;width:7px;background:#c6dfec9e;}
.window:after {content:'';position:absolute;top:50%;width:100%;height:7px;background:#c6dfec9e;}
.floor {position:absolute;bottom:0;width:100%;height:25%;background:linear-gradient(180deg,#283b53,#15253b);border-top:5px solid #506077;}
.sofa {position:absolute;left:36%;bottom:21%;width:125px;height:67px;background:#5b718b;border-radius:20px 20px 7px 7px;border:6px solid #8096ad;box-shadow:0 14px 0 #2a405a;}
.heater-box {position:absolute;right:9%;bottom:18%;width:152px;height:98px;border:6px solid #dce7f1;border-radius:15px;background:repeating-linear-gradient(90deg,#8299b1 0px,#8299b1 9px,#dce4ef 10px,#dce4ef 20px);box-shadow:0 12px 25px #081323;}
.heater-box.active {box-shadow:0 0 64px #ff9e49a8,0 12px 25px #081323;background:repeating-linear-gradient(90deg,#df8945 0px,#df8945 9px,#ffd7a2 10px,#ffd7a2 20px);}
.heat-indicator {position:absolute;right:14%;bottom:54%;font-size:1.55rem;color:#ffd098;letter-spacing:.7rem;animation:float 2s infinite alternate ease-in-out;}
@keyframes float{from{transform:translateY(4px)}to{transform:translateY(-12px)}}
.room-badge {position:absolute;right:20px;top:91px;background:#0c1725be;border:1px solid #7894a76b;border-radius:20px;padding:7px 11px;font-size:.78rem;font-weight:800;}
.note {background:#14263d;border:1px solid #2e4964;border-radius:13px;padding:12px 14px;color:#c3d9eb;font-size:.91rem;margin-top:12px;}
.result {padding:12px 15px;background:#17382f;border:1px solid #346b55;color:#c8f7dc;border-radius:14px;}
hr {border-color:#34506a !important;}
@media (max-width:700px){.hero-title{font-size:1.9rem}.room-scene{min-height:270px}.sofa{left:36%;width:88px}.heater-box{width:108px;height:78px}.window{width:85px;height:100px}}
</style>
""", unsafe_allow_html=True)


def state_room(state: str, action: int | None = None) -> None:
    symbols = {"Cold": "❄️", "Comfortable": "🌿", "Hot": "☀️"}
    descriptions = {"Cold": "Room needs heating", "Comfortable": "Target comfort reached", "Hot": "Room is overheating"}
    heater_on = action == HEATER_ON
    label = "HEATER ON · WARMING" if heater_on else "HEATER OFF · IDLE"
    st.markdown(f"""
    <div class="room-scene {state.lower()}">
      <div class="room-text">{escape(state.upper())}</div>
      <div class="room-sub">{escape(descriptions[state])}</div>
      <div class="room-cloud">{symbols[state]}</div>
      <div class="room-badge">{label}</div>
      <div class="window"></div><div class="floor"></div><div class="sofa"></div>
      <div class="heater-box {'active' if heater_on else ''}"></div>
      {('<div class="heat-indicator">♨ ♨ ♨</div>' if heater_on else '')}
    </div>""", unsafe_allow_html=True)


def q_frame(q: dict) -> pd.DataFrame:
    rows = []
    for state in STATES:
        best = greedy_action(state, q)
        ties = q[state][HEATER_ON] == q[state][HEATER_OFF]
        rows.append({"State": state, "Q(Heater ON)": round(q[state][HEATER_ON], 4),
                     "Q(Heater OFF)": round(q[state][HEATER_OFF], 4),
                     "Learned Action": "Unresolved tie" if ties else ACTION_NAMES[best],
                     "Expected Action": ACTION_NAMES[EXPECTED_POLICY[state]]})
    return pd.DataFrame(rows)


def step_action(action: int, origin: str, learn: bool, alpha: float, gamma: float) -> None:
    if st.session_state.live_steps >= st.session_state.max_steps:
        return
    before = st.session_state.live_state
    after = transition(before, action)
    reward = reward_function(after)
    trace = (update_q(before, action, reward, after, st.session_state.q, alpha, gamma)
             if learn else None)
    st.session_state.live_state = after
    st.session_state.live_last_action = action
    st.session_state.live_steps += 1
    st.session_state.live_total += reward
    st.session_state.live_last_reward = reward
    st.session_state.live_trace = trace
    st.session_state.live_history.append({
        "Step": st.session_state.live_steps, "Before": before,
        "Action": ACTION_NAMES[action], "After": after,
        "Reward": reward, "Source": origin,
        "Q updated?": "Yes" if learn else "No",
    })


def reset_live(state: str = INITIAL_STATE) -> None:
    st.session_state.live_state = state
    st.session_state.live_steps = 0
    st.session_state.live_total = 0
    st.session_state.live_history = []
    st.session_state.live_last_action = None
    st.session_state.live_last_reward = None
    st.session_state.live_trace = None


if "q" not in st.session_state:
    st.session_state.q = blank_q()
    st.session_state.rng = random.Random(42)
    st.session_state.trained_result = None
    st.session_state.eval_result = None
    st.session_state.max_steps = 20
    reset_live()

with st.sidebar:
    st.markdown("### 🎚️ Training controls")
    epsilon = st.slider("Exploration · ε", min_value=0.0, max_value=1.0, value=1.0, step=0.05,
                    help="Initial epsilon = 1.0. Exploration decreases during training.")
    alpha = st.slider("Learning rate · α", min_value=0.05, max_value=1.0, value=0.1, step=0.05)
    gamma = st.slider("Discount factor · γ", min_value=0.0, max_value=0.99, value=0.9, step=0.01)
    episodes = st.number_input("Training episodes", min_value=1, max_value=3000, value=500, step=50)
    steps_per_episode = st.number_input("Steps per episode", min_value=1, max_value=100, value=100, step=1)
    seed = st.number_input("Random seed", min_value=0, max_value=999_999, value=42, step=1)
    train_clicked = st.button("🚀 Train from zero", use_container_width=True, type="primary")
    zero_clicked = st.button("🧹 Reset Q-table", use_container_width=True)
    st.caption(
    "Training settings: α=0.1, γ=0.9, "
    "initial ε=1.0, decay=0.995, min ε=0.01, "
    "500 episodes, 100 steps, seed=42."
)
    st.divider()
    st.markdown("**How it is connected**")
    st.caption("Every decision, state transition, reward and Bellman update is computed by Python functions imported from `heater_rl.py`, extracted/refactored from your notebook. No JavaScript training engine.")

if zero_clicked:
    st.session_state.q = blank_q()
    st.session_state.rng = random.Random(int(seed))
    st.session_state.trained_result = None
    st.session_state.eval_result = None
    st.session_state.max_steps = int(steps_per_episode)
    reset_live()
if train_clicked:
    output = train(
        episodes=int(episodes),
        steps_per_episode=int(steps_per_episode),
        epsilon=epsilon,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        alpha=alpha,
        gamma=gamma,
        seed=int(seed)
    )
    st.session_state.q = output["q"]
    st.session_state.trained_result = output
    st.session_state.eval_result = None
    st.session_state.rng = random.Random(int(seed))
    st.session_state.max_steps = int(steps_per_episode)
    reset_live()
    st.toast("Python Q-Learning training completed successfully!", icon="✅")

st.markdown('<p class="app-eyebrow">REINFORCEMENT LEARNING · LIVE PYTHON ENGINE</p>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">HEATWISE <span style="color:#69e4ce">/</span> Smart Room AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-desc">See how a Q-Learning agent controls a room heater, discovers optimal actions and maximizes long-term reward.</div>', unsafe_allow_html=True)

trained = st.session_state.trained_result is not None
head1, head2, head3, head4 = st.columns(4)
head1.metric("Training", f"{len(st.session_state.trained_result['episode_rewards']) if trained else 0} episodes")
head2.metric("Optimal choices", f"{policy_correct_count(st.session_state.q)}/3")
head3.metric("Greedy score", f"{evaluate(st.session_state.q, steps=int(steps_per_episode))['total']}/{int(steps_per_episode)*10}")
head4.metric("Q-table MAE", f"{q_mae(st.session_state.q, gamma=(st.session_state.trained_result['config']['gamma'] if trained else gamma)):.3f}")

if trained:
    cfg = st.session_state.trained_result["config"]
    if (cfg["epsilon"] != epsilon or cfg["alpha"] != alpha or cfg["gamma"] != gamma or
        cfg["episodes"] != episodes or cfg["steps_per_episode"] != steps_per_episode or cfg["seed"] != seed):
        st.warning("Sidebar settings have changed since last training. Q-table still reflects the *previous* training settings; click **Train from zero** to apply changes.")
else:
    st.info("Start by demonstrating a manual Heater ON/OFF decision. Then click **Train from zero** in the sidebar to let the actual Python agent learn.")

tab_sim, tab_train, tab_eval, tab_explain = st.tabs([
    "🏠 Live Room Simulation", "📈 Training Results", "🎯 Greedy Evaluation", "🧠 Explain the Algorithm",
])

with tab_sim:
    left, right = st.columns([1.4, 1], gap="large")
    with right:
        st.subheader("🎮 Control the environment")
        desired_state = st.selectbox("Start / restart from", list(STATES), index=0, key="start_state")
        if st.button("↺ Reset room (keep learned Q-values)", use_container_width=True):
            reset_live(desired_state)
        learn_live = st.checkbox("Apply Bellman update on each live step", value=False,
                                 help="Unchecked = simulation without learning. Checked = actions update the actual Q-table.")
        st.markdown("**Choose who controls the heater**")
        cc1, cc2 = st.columns(2)
        is_over = st.session_state.live_steps >= st.session_state.max_steps
        with cc1:
            if st.button("🔥 Heater ON", disabled=is_over, use_container_width=True):
                step_action(HEATER_ON, "Manual", learn_live, alpha, gamma)
        with cc2:
            if st.button("🧊 Heater OFF", disabled=is_over, use_container_width=True):
                step_action(HEATER_OFF, "Manual", learn_live, alpha, gamma)
        if st.button("🤖 Agent · Greedy action", disabled=is_over, use_container_width=True, type="primary"):
            a = greedy_action(st.session_state.live_state, st.session_state.q)
            step_action(a, "Greedy (no exploration)", learn_live, alpha, gamma)
        if st.button("🎲 Agent · ε-Greedy action", disabled=is_over, use_container_width=True):
            a = choose_action(st.session_state.live_state, st.session_state.q, epsilon, st.session_state.rng)
            step_action(a, f"Epsilon greedy (ε={epsilon:.2f})", learn_live, alpha, gamma)
        if is_over:
            st.warning("Episode finished. Reset the room to start a new episode; the trained Q-table is kept.")
        st.markdown('<div class="note">📌 <b>Manual</b> buttons allow you to demonstrate the environment before training. <b>Greedy</b> chooses the highest Q-value. <b>ε-greedy</b> occasionally explores. Turn on the checkbox to see live Bellman updates.</div>', unsafe_allow_html=True)
    with left:
        state_room(st.session_state.live_state, st.session_state.live_last_action)
        m1, m2, m3 = st.columns(3)
        m1.metric("Live steps", f"{st.session_state.live_steps}/{st.session_state.max_steps}")
        m2.metric("Last reward", "—" if st.session_state.live_last_reward is None else f"{st.session_state.live_last_reward:+d}")
        m3.metric("Episode reward", st.session_state.live_total)
        if st.session_state.live_history:
            latest = st.session_state.live_history[-1]
            st.markdown(f'<div class="result">{escape(latest["Before"])} + {escape(latest["Action"])} → <b>{escape(latest["After"])}</b> · Reward <b>{latest["Reward"]:+d}</b> · {escape(latest["Source"])}</div>', unsafe_allow_html=True)
        if st.session_state.live_trace:
            u = st.session_state.live_trace
            with st.expander("🔍 Actual Bellman update for the latest step", expanded=True):
                st.latex(r"Q_{new}(s,a)=Q_{old}(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q_{old}(s,a)]")
                st.code(f"{u['old_q']:.4f} + {u['alpha']:.2f} × [{u['reward']:+d} + {u['gamma']:.2f} × {u['best_next_q']:.4f} − {u['old_q']:.4f}] = {u['new_q']:.4f}", language="text")
    st.divider()
    l_table, l_log = st.columns([1, 1.5])
    with l_table:
        st.subheader("Current Q-table")
        st.dataframe(q_frame(st.session_state.q), hide_index=True, use_container_width=True)
    with l_log:
        st.subheader("Live decision history")
        if st.session_state.live_history:
            st.dataframe(pd.DataFrame(st.session_state.live_history).iloc[::-1], hide_index=True, use_container_width=True)
        else:
            st.caption("No interactions yet. Use any action button to generate real transition logs.")

with tab_train:
    st.subheader("What did the agent learn?")
    if not trained:
        st.info("Press **Train from zero** in the left sidebar. The displayed curves and Q-values will come from Python training, not pre-generated sample data.")
    else:
        result = st.session_state.trained_result
        rewards = result["episode_rewards"]
        window = min(20, len(rewards))
        graph = pd.DataFrame({
            "Episode": range(1, len(rewards) + 1),
            "Actual training reward": rewards,
            f"{window}-episode rolling average": pd.Series(rewards).rolling(window, min_periods=1).mean(),
            "Theoretical episode ceiling": [result["config"]["steps_per_episode"] * 10] * len(rewards),
        }).set_index("Episode")
        st.line_chart(graph, height=310)
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg first 50 episodes", f"{sum(rewards[:50])/min(50,len(rewards)):.2f}")
        c2.metric("Avg last 50 episodes", f"{sum(rewards[-50:])/min(50,len(rewards)):.2f}")
        c3.metric("Final cumulative reward", f"{result['cumulative_rewards'][-1]:,}")
        st.caption("Training rewards may dip even after the optimal policy is learned: ε-greedy exploration intentionally takes some non-greedy actions. Use the separate greedy evaluation for policy quality.")
        st.markdown("#### Learned Q-values vs theoretical Q* values")
        target = theoretical_q(result["config"]["gamma"])
        vals = []
        for s in STATES:
            for a in ACTIONS:
                vals.append({"State": s, "Action": ACTION_NAMES[a],
                             "Learned Q": round(result["q"][s][a], 3),
                             "Optimal Q*": round(target[s][a], 3),
                             "Absolute difference": round(abs(result["q"][s][a] - target[s][a]), 3)})
        st.dataframe(pd.DataFrame(vals), hide_index=True, use_container_width=True)
        st.caption("A correct greedy policy is not the same as perfectly converged Q-values. Some state-actions may have fewer updates, particularly at low exploration rates.")
        st.markdown("#### Cumulative reward")
        st.line_chart(pd.DataFrame({"Cumulative reward": result["cumulative_rewards"]},
                                   index=range(1, len(rewards) + 1)), height=220)

    st.divider()
    st.subheader("🔬 Experiment: ε = 0.1 vs ε = 0.2")
    st.caption("Runs two independent Python training experiments from zero with the same seed and other hyperparameters. The comparison is controlled, not a guaranteed improvement.")
    if st.button("Compare both epsilon values", use_container_width=False):
        trials = []
        for e in (0.1, 0.2):
            r = train(episodes=int(episodes), steps_per_episode=int(steps_per_episode),
                      epsilon=e, alpha=alpha, gamma=gamma, seed=int(seed))
            vals = r["episode_rewards"]
            trials.append({"ε": e, "Avg last 50 (training)": round(sum(vals[-50:])/min(50,len(vals)), 2),
                           "Q-table MAE": round(q_mae(r["q"], gamma), 4),
                           "Greedy reward": evaluate(r["q"], steps=int(steps_per_episode))["total"],
                           "Correct decisions": f"{policy_correct_count(r['q'])}/3"})
        st.session_state.comparison = trials
    if "comparison" in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state.comparison), hide_index=True, use_container_width=True)

with tab_eval:
    st.subheader("Greedy policy: test without exploration")
    start = st.selectbox("Evaluation initial state", list(STATES), index=0, key="eval_start")
    eval_steps = st.slider("Evaluation horizon (steps)", 1, 50, 20, key="eval_steps")
    if st.button("▶ Run greedy evaluation", type="primary"):
        st.session_state.eval_result = evaluate(st.session_state.q, start_state=start, steps=eval_steps)
    if st.session_state.eval_result:
        ev = st.session_state.eval_result
        score, target_col, completed_col = st.columns(3)
        score.metric("Actual score", ev["total"])
        target_col.metric("Maximum possible", ev["maximum"])
        completed_col.metric("Steps", len(ev["timeline"]))
        if ev["total"] == ev["maximum"]:
            st.success("Maximum score achieved under this environment's reward rules.")
        else:
            st.warning("Score below theoretical maximum. Train the agent, then evaluate again.")
        chosen = st.slider("🔎 Replay the agent's decisions", 1, len(ev["timeline"]), 1)
        frame = ev["timeline"][chosen-1]
        a = HEATER_ON if frame["action"] == ACTION_NAMES[HEATER_ON] else HEATER_OFF
        demo_col, details_col = st.columns([1.35, 1])
        with demo_col:
            state_room(frame["next_state"], a)
        with details_col:
            st.markdown("#### Decision walkthrough")
            st.write(f"**Step:** {chosen} / {len(ev['timeline'])}")
            st.write(f"**Current state:** {frame['state']}")
            st.write(f"**Chosen action:** {frame['action']}")
            st.write(f"**Next state:** {frame['next_state']}")
            st.write(f"**Immediate reward:** {frame['reward']:+d}")
            st.write(f"**Total reward so far:** {frame['total']}")
        st.dataframe(pd.DataFrame(ev["timeline"]), hide_index=True, use_container_width=True)
    else:
        st.info("Click **Run greedy evaluation** to generate a full Python rollout and inspect each step using the replay slider.")
    st.caption("Important: evaluation only chooses argmax Q and calculates rewards. It does NOT update the Q-table, and ignores epsilon.")

with tab_explain:
    st.subheader("A clear explanation for your doctor / TA")
    st.markdown("""
**Problem:** The agent learns when to turn a heater **ON** or **OFF** to reach and maintain the **Comfortable** state.

**States:** `Cold`, `Comfortable`, `Hot`.

**Actions:** `Heater ON (0)`, `Heater OFF (1)`.

**Reward of next state:** `Comfortable = +10`, `Cold = −1`, `Hot = −5`.

**Real Q-Learning:** Python selects an action using epsilon-greedy, uses `transition()` to get the next state, calls `reward_function()` and updates the Q-table via Bellman Q-learning. The app imports these functions from `heater_rl.py`; it does not execute a separate JavaScript model.
""")
    st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha\Big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\Big]")
    st.markdown("""
**Expected optimal policy:**
- `Cold → Heater ON → Comfortable (+10)`
- `Comfortable → Heater OFF → Comfortable (+10)`
- `Hot → Heater OFF → Comfortable (+10)`

**Why is Q* = 100 while the 20-step reward is 200?**
The 20-step score is an **undiscounted** finite sum of rewards: `20 × 10 = 200`. At `γ = 0.9`, the infinite-horizon discounted return for a continuing environment is `10 / (1 − 0.9) = 100`. The 20-step cap is only a truncation used for simulation; it is **not** treated as a terminal state during Q-learning updates.

**Limitations:** This is a discrete, deterministic teaching environment, not a physical thermal model or a real temperature sensor. Random exploration causes training scores to fluctuate. Q-values may still be below theoretical convergence after 500 episodes even with the correct policy.
""")
    st.code("from heater_rl import transition, reward_function, choose_action, update_q, train, evaluate", language="python")

st.markdown("<div style='color:#839bb7;text-align:center;padding-top:25px;font-size:.8rem'>HEATWISE · Student demonstration · Executing the Python Q-Learning engine</div>", unsafe_allow_html=True)
