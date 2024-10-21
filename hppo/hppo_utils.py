import torch
import scipy.signal
import numpy as np
import torch.nn as nn



def get_mean_and_standard_deviation_difference(results):
    """
    From a list of lists of specific agent results it extracts the mean result and the mean result plus or minus
    some multiple of standard deviation.
    :param results:
    :return:
    """

    def get_results_at_a_time_step(results_, timestep):
        results_at_a_time_step = [result[timestep] for result in results_]
        return results_at_a_time_step

    def get_std_at_a_time_step(results_, timestep):
        results_at_a_time_step = [result[timestep] for result in results_]
        return np.std(results_at_a_time_step)

    mean_results = [np.mean(get_results_at_a_time_step(results, timestep)) for timestep in range(len(results[0]))]
    mean_minus_x_std = [mean_val - get_std_at_a_time_step(results, timestep)
                        for timestep, mean_val in enumerate(mean_results)]
    mean_plus_x_std = [mean_val + get_std_at_a_time_step(results, timestep)
                       for timestep, mean_val in enumerate(mean_results)]
    return mean_minus_x_std, mean_results, mean_plus_x_std


def get_y_limits(results):
    """
    Extracts the minimum and maximum seen y_vals from a set of results.
    :param results:
    :return:
    """
    res_flattened = np.array(results).flatten()
    max_res = np.max(res_flattened)
    min_res = np.min(res_flattened)
    y_limits = [min_res - 0.05 * (max_res - min_res), max_res + 0.05 * (max_res - min_res)]

    return y_limits



def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def extract_rolling_score(record_mark, action_space_pattern, worker_idx):
    indicator = 0
    for idx in worker_idx:
        raw_delay = np.load('data/rolling_data/{}/{}/delay_{}.npz'.format(record_mark, action_space_pattern, idx))
        raw_queue = np.load('data/rolling_data/{}/{}/queue_{}.npz'.format(record_mark, action_space_pattern, idx))
        if indicator == 0:
            delay = np.array([raw_delay['delay']])
            queue = np.array([raw_queue['queue']])
            indicator = 1
        else:
            delay = np.append(delay, [raw_delay['delay']], axis=0)
            queue = np.append(queue, [raw_queue['queue']], axis=0)
    return delay, queue, delay.shape[1]


def extract_over_all_rs(record_mark, worker_idx: list):
    delay_con, queue_con, size_con = extract_rolling_score(record_mark, 'continuous', worker_idx)
    delay_dis, queue_dis, size_dis = extract_rolling_score(record_mark, 'discrete', worker_idx)
    delay_hybrid, queue_hybrid, size_hybrid = extract_rolling_score(record_mark, 'hybrid', worker_idx)
    min_size = min(size_con, size_dis, size_hybrid)
    delay = np.array([
        delay_con[:, :min_size],
        delay_dis[:, :min_size],
        delay_hybrid[:, :min_size]
    ])

    queue = np.array([
        queue_con[:, :min_size],
        queue_dis[:, :min_size],
        queue_hybrid[:, :min_size]
    ])

    return delay, queue


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)
        

def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

    Advantage: Less memory requirement O(M*d + N*d + M*N) instead of O(N*M*d)
    Computationally more expensive? Maybe, Not sure.
    adapted from: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    '''
    # print("x",x)
    # print("y",y)

    x_norm = (x ** 2).sum(1).view(-1, 1)   #sum(1)将一个矩阵的每一行向量相加
    y_norm = (y ** 2).sum(1).view(1, -1)
    # print("x_norm",x_norm)
    # print("y_norm",y_norm)
    y_t = torch.transpose(y, 0, 1)  #交换一个tensor的两个维度
    # a^2 + b^2 - 2ab
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)    #torch.mm 矩阵a和b矩阵相乘
    # dist[dist != dist] = 0 # replace nan values with 0
    # print("dist",dist)
    return dist




        
        

