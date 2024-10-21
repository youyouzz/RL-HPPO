## result

### 0

settings: simple wet env + with action mask + hybrid action space + ppo + no random initial input

platform: AMD 7950X + Nvidia RTX 4090

training episode: 2000

training time: 60s

version: 20240508041500

![trainning result](../log\RTS-T2-20240508041500\total_reward_history.png "RTS-T2-20240508041500")

### 1

settings: simple wet env + with action mask + hybrid action space + ppo + no random initial input

platform: AMD 7950X + Nvidia RTX 4090

training episode: 3000

eps_clip = 0.2

version: 20240508041500

![trainning result](../log\RTS-T2-20240508041500_1\total_reward_history.png "RTS-T2-20240508041500")

### 2

settings: simple wet env + with action mask + hybrid action space + ppo + no random initial input

platform: AMD 7950X + Nvidia RTX 4090

training episode: 3000

eps_clip = 0.25

version: 20240508041500

![trainning result](../log\RTS-T2-20240508041500_2\total_reward_history.png "RTS-T2-20240508041500")

### 3

settings: simple wet env + with action mask + hybrid action space + ppo + no random initial input

platform: AMD 7950X + Nvidia RTX 4090

training episode: 3000

eps_clip = 0.15

version: 20240508041500

![trainning result](../log\RTS-T2-20240508041500_3\total_reward_history.png "RTS-T2-20240508041500")

## compare with gurobi

get no empty last_opt_wet_df:20240507164500 的plan是 20240507161500的gurobi计算结果，即半个小时之前的版本的最优结果

### 1

add gurobi opt plan

lack of: prioritized replay buffer, add gaussian action noise

![trainning result](../log\RTS-T2-20240507161500_1\total_reward_history.png "RTS-T2-20240507161500")

### 2

add gurobi opt plan

lack of: prioritized replay buffer, add gaussian action noise

fix: Be consistent with gurobi's settings self.period_list=[1,24], otherwise data reading will cause errors

![trainning result](../log\RTS-T2-20240507161500_2\total_reward_history.png "RTS-T2-20240507161500")

### 3

add gurobi opt plan

lack of: prioritized replay buffer, add gaussian action noise

fix action mask error

![trainning result](../log\RTS-T2-20240507161500_3\total_reward_history.png "RTS-T2-20240507161500")

### 4

add gurobi opt plan

lack of: prioritized replay buffer, add gaussian action noise

fix action mask error and change the random seed to 1

![trainning result](../log\RTS-T2-20240507161500_4\total_reward_history.png "RTS-T2-20240507161500")

### 5


![trainning result](../log\RTS-T2-20240507161500_4\total_reward_history.png "RTS-T2-20240507161500")

### 6


![trainning result](../log\RTS-T2-20240507161500_4\total_reward_history.png "RTS-T2-20240507161500")

### 7


![trainning result](../log\RTS-T2-20240507161500_4\total_reward_history.png "RTS-T2-20240507161500")
