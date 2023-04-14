"""
Deep Q Network off-policy
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(2)


class Network(nn.Module):
    """
    Network Structure
    """
    def __init__(self,
                 n_features,
                 n_actions,
                 n_neuron=10
                 ):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_neuron, bias=True),
            nn.Linear(in_features=n_neuron, out_features=n_actions, bias=True),
            nn.ReLU()
        )

    def forward(self, s):
        """

        :param s: s
        :return: q
        """
        q = self.net(s)
        return q


class DeepQNetwork(nn.Module):
    """
    Q Learning Algorithm
    """
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None):
        super(DeepQNetwork, self).__init__()

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        # 这里用pd.DataFrame创建的表格作为memory
        # 表格的行数是memory的大小，也就是transition的个数
        # 表格的列数是transition的长度，一个transition包含[s, a, r, s_]，其中a和r分别是一个数字，s和s_的长度分别是n_features
        self.memory = pd.DataFrame(np.zeros((self.memory_size, self.n_features*2+2)))

        # build two network: eval_net and target_net
        self.eval_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        self.target_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        # 记录每一步的误差
        self.cost_his = []


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            # hasattr用于判断对象是否包含对应的属性。
            self.memory_counter = 0

        transition = np.hstack((s, [a,r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory.iloc[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            s = torch.FloatTensor(observation)
            actions_value = self.eval_net(s)
            action = [np.argmax(actions_value.detach().numpy())][0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        # 复制网络参数
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget params replaced\n')

        # sample batch memory from all memory
        batch_memory = self.memory.sample(self.batch_size) \
            if self.memory_counter > self.memory_size \
            else self.memory.iloc[:self.memory_counter].sample(self.batch_size, replace=True)

        # run the nextwork
        s = torch.FloatTensor(batch_memory.iloc[:, :self.n_features].values)
        s_ = torch.FloatTensor(batch_memory.iloc[:, -self.n_features:].values)
        q_eval = self.eval_net(s)
        q_next = self.target_net(s_)

        # change q_target w.r.t q_eval's action
        q_target = q_eval.clone()

        # 更新值
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory.iloc[:, self.n_features].values.astype(int)
        reward = batch_memory.iloc[:, self.n_features + 1].values

        q_target[batch_index, eval_act_index] = torch.FloatTensor(reward) + self.gamma * q_next.max(dim=1).values

        # train eval network
        loss = self.loss_function(q_target, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.detach().numpy())

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.show()

































































































