import numpy as np


def cosine_scheduler(base_value, final_value, episodes, niter_per_ep, warmup_episodes=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_episodes * niter_per_ep
    if warmup_episodes > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(episodes * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == episodes * niter_per_ep
    return schedule
