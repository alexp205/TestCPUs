import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import gym

debug = False

load_model = False
visualize = True
test_model = False

class DQNAgent:
    def __init__(self, s_size, a_size):
        self.s_size = s_size
        self.a_size = a_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.95
        if not load_model:
            self.epsilon = 1.0
        else:
            self.epsilon = 0.01
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.001
        self.model = self.build()

    def build(self):
        model = models.Sequential()

        model.add(layers.Dense(24, input_dim=self.s_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.a_size, activation='linear'))

        # DEBUG
        if debug:
            model.summary()
            #input("wait")

        model.compile(loss='mse', optimizer='adam')

        return model

    def remember(self, s, a, r, s_prime, done):
        self.memory.append((s, a, r, s_prime, done))

    def act(self, s):
        if not load_model and not test_model:
            if np.random.rand() <= self.epsilon:

                # DEBUG
                if debug: print("exploring!")

                return random.randrange(self.a_size)

        q_vals = self.model.predict(s)
        return np.argmax(q_vals[0])

    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        for s, a, r, s_prime, done in batch:
            target = r
            if not done:
                target = (r + self.gamma * np.amax(self.model.predict(s_prime)[0]))
            target_f = self.model.predict(s)
            target_f[0][a] = target
            self.model.fit(s, target_f, verbose=0)
        if not load_model and not test_model:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def load(self, f):
        print("loading model!")
        self.model.load_weights(f)

    def save(self, f):
        print("saving model!")
        self.model.save_weights(f)

def main():
    env = gym.make("CartPole-v1")
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    agent = DQNAgent(s_size, a_size)
    if load_model:
        agent.load("./other_save/cartpole_dqn.h5")

    # some vars
    batch_size = 32
    mem_stored = 0
    r_sum = 0
    total_r = None
    s = env.reset()

    run = 0
    while True:
        if visualize:
            env.render()

        # preprocess
        s = np.reshape(s, [1, s_size])

        # get action
        a = agent.act(s)

        # DEBUG
        if debug:
            for layer in agent.model.layers: print(layer.get_config(), layer.get_weights())
            print("action: {}".format(a))
            input("wait")

        s_prime, r, done, _ = env.step(a)
        if not test_model:
            r = r if not done else -r
            r_sum += r
            s_prime = np.reshape(s_prime, [1, s_size])
            agent.remember(s, a, r, s_prime, done)
            mem_stored += 1
            s = s_prime

            if mem_stored > batch_size:
                agent.replay(batch_size)
                mem_stored = 0

            if done:
                run += 1

                if 0 == run % 100:
                    agent.save("./other_save/cartpole_dqn.h5")

                total_r = r_sum if total_r is None else total_r * 0.95 + r_sum * 0.05 # scaled off of gamma
                print("At run #{}".format(run))
                print("resetting env! episode reward total was {} w/ running mean {}".format(r_sum, total_r))
                print("current exploration constant: {:.2}".format(agent.epsilon))
                print()

                r_sum = 0
                s = env.reset()
        else:
            r = r if not done else -r
            r_sum += r
            s = s_prime

            if done:
                print("balanced for {} steps!".format(r_sum))

                r_sum = 0
                s = env.reset()

if __name__ == "__main__":
    main()
