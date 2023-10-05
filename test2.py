import unittest
import numpy as np

def boltzmann_policy(x, tau):
    """ Returns softmax probabilities with temperature tau
        Input:  x -- 1-dimensional array
        Output: idx -- chosen index
    """
    # ----------------------------------------------

    x_temp = np.array(x) / tau
    e_x = np.exp(x_temp)
    probs = e_x / np.sum(e_x)

    # sample according to probs
    idx = np.random.choice(np.arange(len(x)), p=probs)

    idx = np.random.choice(np.flatnonzero(probs == probs.max()))
    idx = np.argmax(probs)

    # ----------------------------------------------
    return idx


def gradient_bandit(
        bandit: Bandit,
        alpha: float,
        use_baseline: bool = True,
) -> Tuple[list, list, list]:
    """
    .inputs:
      bandit: A bandit problem, instantiated from the above class.
      alpha: The learning rate.
      use_baseline: Whether or not use avg return as baseline.
    .outputs:
      rew_record: The record of rewards at each timestep.
      avg_ret_record: The average summation of rewards up to step t, where t goes from 0 to n_pulls. For example: If
      we define `ret_T` = \sum^T_{t=0}{r_t}, `avg_ret_record` = ret_T / (1+T).
      tot_reg_record: The  regret up to step t, where t goes from 0 to n_pulls.
      opt_action_perc_record: Percentage of optimal arm selected.
    """
    # init h (the logits)
    h = np.array([0] * bandit.n_arm, dtype=float)

    r_bar_t = 0
    ret = .0  # total reward
    rew_record = []  # reward at each time step
    avg_ret_record = []  # avg reward up to time T
    tot_reg_record = []  # regret at each time step
    optimal_action_selection_record = []  # binary value if opt action selected
    opt_action_perc_record = []  # convert ^ to percentage

    # ----------------------------------------------

    p_true = np.array(bandit.actual_toxicity_prob)
    true_expected_rewards = - np.abs(bandit.theta - p_true)
    true_optimal_arm_idx = np.argmax(true_expected_rewards)
    true_optimal_reward = max(true_expected_rewards.tolist())

    # initialize preference
    h = np.zeros(bandit.n_arm)

    for t in range(bandit.n_pulls):
        # Select action
        probs = softmax(h)
        action_idx = np.random.choice(np.arange(bandit.n_arm), p=probs)

        # Get reward
        rew = bandit.pull(action_idx)

        # Update preference (for action taken)
        h[action_idx] = h[action_idx] + alpha * (rew - r_bar_t)(1 - probs[action_idx])

        # update preference (for all other actions)
        for i in range(bandit.n_arm):
            if i != action_idx:
                h[i] = h[i] - alpha * (rew - r_bar_t) * probs[i]
            else:
                h[i] = h[i] + alpha * (rew - r_bar_t) * (1 - probs[i])

        # update baseline
        if use_baseline and t > 0:
            r_bar_t = r_bar_t + (1 / t) * (rew - r_bar_t)

        ret += rew  # total reward
        rew_record.append(rew)
        avg_ret_record.append(ret / t)  # avg reward up to time T

        regret = t * true_optimal_reward - ret
        tot_reg_record.append(regret)
        optimal_action_selection_record.append(1 if action_idx == true_optimal_arm_idx else 0)

    # cumsum
    opt_action_perc_record = np.cumsum(optimal_action_selection_record) / len(optimal_action_selection_record)

    # ----------------------------------------------

    return rew_record, avg_ret_record, tot_reg_record, opt_action_perc_record
if __name__ == '__main__':
    unittest.main()
