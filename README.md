# WET-RL

## 1 introduction

Hybrid Proximal Policy Optimization (HPPO) for WetEnv

This project implements a Hybrid Proximal Policy Optimization (HPPO) algorithm for reinforcement learning in a custom environment called WetEnv. The HPPO algorithm is designed to handle both discrete and continuous action spaces.

## 2 env & model

### 2.1 Project Structure

-`env/wet_rl_env.py`: Custom environment implementation (WetEnv)

-`hppo/hppo.py`: HPPO algorithm implementation, including PPO_Hybrid, ActorCritic_Hybrid, and PPOBuffer classes

-`hppoTrainer.py`: Trainer class for managing the training process

-`util/`: Utility functions and configurations

### 2.2 Key Components

**You can modify the hyperparameters in the `hppoTrainer.py` file or pass them as command-line arguments.**

1. **WetEnv**: Custom environment with hybrid action space (discrete and continuous).
2. **PPO_Hybrid**: Implementation of the Hybrid Proximal Policy Optimization algorithm.
3. **ActorCritic_Hybrid**: Neural network architecture for the actor-critic model.
4. **PPOBuffer**: Experience replay buffer for storing and sampling transitions.
5. **Trainer**: Manages the training process, including episode collection, agent updates, and logging.

**Key hyperparameters include:**

- Learning rates (actor, critic, std)
- Discount factor (gamma)
- GAE lambda
- Clipping epsilon
- Target KL divergence
- Entropy coefficient

Refer to the `hppoTrainer.py` file for a complete list of hyperparameters and their default values.

**Results**

After training, the results will be saved in the `log/` directory, including:

- Total reward history (`.npy` file)
- Total reward plot (`.png` file)

## 4 Experiment Design

1. 调参对比，曲线优化
2. evaluate function design1： 用当前版本input直接计算，用下个版本的last opt plan作为对比
3. evaluate function design2： 用当前版本input训练，生成随机初始状态，用下个版本的input做推理，用下下个版本的last opt plan作为对比，查看模型泛化能力
4. 单个版本的与gurobi结果的对比，要gantt结果可视化
5. 大量版本的与gurobi结果的对比，对比可视化，不需要gantt结果可视化
6. 计算时间与episode的曲线![trainning result](./log/RTS-T2-20240507161500_8/total_reward_history.png "RTS-T2-20240507161500")


## 5 todo list

- [X] pm down 放入 observation space
- [X] constraints referenced from gurobi ppt(one by one)
- [X] step update/reward: 酸浓度更新不能超过上限
- [X] step update/reward: 酸浓度更新如果超过上限 下一步换酸还是现在换酸 还是用reward去调节使得完美利用酸浓度
- [X] step update/action mask: 酸浓度超3402上限    →    当前Period不能加工3402
- [X] step update/action mask: 酸寿命超3402上限    →    当前Period不能加工3402
- [X] action mask/reward: 必须达到酸浓度或者算寿命上限才能换酸，否则不能换酸
- [X] reward function
- [X] 如果没有初始酸浓度，则给予300浓度
- [ ] acid alter 没有酸浓度上升数据
- [ ] gurobi的酸浓度上限显示5500或5000，与model读入的数据分析不一致
- [ ] 不需要move量作为action，修改decoder，调整前6h的限制

## 6 Issues to be optimized

1. 均衡性，计算平均值时，gurobi模型将pm和down机的也算进来了，应该去掉
2. 全局step均衡性，rl改为了同一时间内的step均衡性
3. 计算均衡性，是否应该使用归一化方差
4. 目标函数的参数设计应该根据结果再进行优化调整
5. 

## 7 Important Issues

1. buffer size can not bigger than 64, because the buffer size must < the ptr size
2. Be consistent with gurobi's settings self.period_list=[1,24], otherwise data reading will cause errors
3. needn't use continuous action, the move qauntity is set to be max directly

## 8 reference

1. https://github.com/ray-project/ray
2. https://docs.ray.io/en/latest/ray-overview/index.html
3. https://docs.ray.io/en/latest/rllib/index.html
4. [ray-rllib 开发者文档总结 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/432426803)
5. [Metro1998/hppo-in-traffic-signal-control (github.com)](https://github.com/Metro1998/hppo-in-traffic-signal-control/tree/main)
6. https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo
7. https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py
8. 深度强化学习调参技巧：以D3QN、TD3、PPO、SAC算法为例（有空再添加图片） - 曾伊言的文章 - 知乎
   https://zhuanlan.zhihu.com/p/345353294
9.
