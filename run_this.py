from maze_env import Maze
from RL_brain import DeepQNetwork

def run_maze():
    step = 0  # 为了记录走到第几步，记忆录中积累经验（也就是积累一些transition）之后再开始学习
    for episode in range(200):
        # initial observation
        observation = env.reset()

        while True:
            # refresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # !! restore transition
            RL.store_transition(observation, action, reward, observation_)

            # 超过200条transition之后每隔5步学习一次
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print("game over")
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000)
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()


