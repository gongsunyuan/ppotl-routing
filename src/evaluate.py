import numpy as np


def evaluate_policy(agent, env, episodes, aug=None):
    delays, losses, rewards = [], [], []
    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_r, ep_d, ep_l, n = 0.0, 0.0, 0.0, 0
        while not done:
            st = aug(s, env) if aug else s
            a, _ = agent.select_action(st, greedy=True)
            s, r, done, info = env.step(a)
            ep_r += r
            ep_d += info["delay"]
            ep_l += info["loss"]
            n += 1
        rewards.append(ep_r / max(n, 1))
        delays.append(ep_d / max(n, 1))
        losses.append(ep_l / max(n, 1))
    return {"reward": float(np.mean(rewards)), "delay": float(np.mean(delays)),
            "loss": float(np.mean(losses)),
            "delay_std": float(np.std(delays)), "loss_std": float(np.std(losses))}


def convergence_episode(rewards, window=10, threshold=0.95):
    if len(rewards) < window:
        return len(rewards)
    arr = np.asarray(rewards)
    smooth = np.convolve(arr, np.ones(window) / window, mode="valid")
    target = threshold * smooth[-1]
    for i, v in enumerate(smooth):
        if v >= target:
            return int(i + window)
    return len(arr)
