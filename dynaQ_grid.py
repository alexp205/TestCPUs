import random
import pickle
import gym_minigrid
import gym
from gym_minigrid.wrappers import *
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from sklearn import linear_model

debug = False

load_model = True
visualize = True
test_model = False

# the Dyna-Q agent class
class DynaQ_Agent:
    def __init__(self, gamma, alpha, epsilon, epsilon_decay, num_replay):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.1
        self.num_replay = num_replay
        self.memory = deque(maxlen=3000)
        self.Qval_fxn_approx = self.build_approx()
        self.is_fit = False
        self.model = self.build_model()
        self.explore_mgr = 0

    # build lin model for Q val fxn approx
    def build_approx(self):
        Qval_fxn_approx = linear_model.SGDRegressor()

        # DEBUG
        if debug:
            print(Qval_fxn_approx)
            input("wait")

        return Qval_fxn_approx

    # build NN for model
    def build_model(self):
        model = models.Sequential()

        model.add(layers.Dense(16, input_dim=82, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(82, activation='linear'))

        # DEBUG
        if debug:
            model.summary()
            input("wait")

        model.compile(loss='mse', optimizer='adam')

        return model

    def act(self, s):
        if not test_model:
            if random.random() < self.epsilon:

                # DEBUG
                if debug: print("exploring!")

                cand_a = random.randrange(6)
                if self.is_fit:
                    s_a_vec = np.concatenate((s, [cand_a]))
                    cand_Q = self.Qval_fxn_approx.predict([s_a_vec])
                else:
                    cand_Q = 0

                return cand_a, cand_Q

        cand_Q = None
        next_Q = []
        for act in range(6):
            s_act_vec = np.concatenate((s, [act]))
            next_Q.append(self.Qval_fxn_approx.predict([s_act_vec]))
        cand_a = np.argmax(next_Q)
        cand_Q = max(next_Q)

        return cand_a, cand_Q

    def update(self, s, a, r, s_prime, done, curr_dir):
        self.explore_mgr += 1

        s_a_vec = np.concatenate((s, [a]))
        s_a_vec_input = np.atleast_2d(s_a_vec)
        if self.is_fit:
            curr_Q = self.Qval_fxn_approx.predict([s_a_vec])
        else:
            curr_Q = 0
        _, best_next_Q = self.act(s_prime)

        Q_update = curr_Q + self.alpha * (r + self.gamma * best_next_Q - curr_Q)
        if self.is_fit:
            self.Qval_fxn_approx.partial_fit([s_a_vec], np.array([Q_update]).ravel())
        else:
            self.Qval_fxn_approx.fit([s_a_vec], np.array([Q_update]).ravel())
            self.is_fit = True

        s_prime_r_vec = np.concatenate((s_prime, [r]))
        s_prime_r_vec_input = np.atleast_2d(s_prime_r_vec)
        self.model.fit(s_a_vec_input, s_prime_r_vec_input, verbose=0)

        self.memory.append((s, a))
        for n in range(self.num_replay):
            sample_s, sample_a = random.sample(self.memory, 1)[0]
            sample_s_a_vec = np.concatenate((sample_s, [sample_a]))
            sample_s_a_vec_input = np.atleast_2d(sample_s_a_vec)
            sample_s_prime_r_vec = self.model.predict(sample_s_a_vec_input)[0]
            sample_r = sample_s_prime_r_vec[-1]
            sample_s_prime = sample_s_prime_r_vec[:-1]
            sample_Q = self.Qval_fxn_approx.predict([sample_s_a_vec])
            _, sample_best_next_Q = self.act(sample_s_prime)
            sample_Q_update = sample_Q + self.alpha * (sample_r + self.gamma * sample_best_next_Q - sample_Q)
            self.Qval_fxn_approx.partial_fit([sample_s_a_vec], np.array([sample_Q_update]).ravel())

        if not test_model:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
        if 0 == self.explore_mgr%100:
            print("current exploration constant: {}".format(self.epsilon))

    def load(self, f):
        print("loading model!")
        self.Qval_fxn_approx = pickle.load(open("./grid_save/grid_qval_dynaq.p", "rb"))
        self.model.load_weights(f)

    def save(self, f):
        print("saving model!")
        pickle.dump(self.Qval_fxn_approx, open("./grid_save/grid_qval_dynaq.p", "wb"))
        self.model.save_weights(f)

def preprocess(img):
    proc_img = None

    proc_img = img
    proc_img = np.sum(proc_img, axis=2)
    proc_img
    proc_img[proc_img == 7] = 0
    proc_img = proc_img.ravel()

    # DEBUG
    if debug:
        print(img)
        print(img.shape)
        print(proc_img)
        input("wait")

    return proc_img

"""
# Turn left, turn right, move forward
left = 0
right = 1
forward = 2

# Pick up an object
pickup = 3
# Drop an object
drop = 4
# Toggle/activate an object
toggle = 5

# Done completing task
done = 6
"""
def get_action(num):
    if 0 == num:
        return env.actions.left
    elif 1 == num:
        return env.actions.right
    elif 2 == num:
        return env.actions.forward
    elif 3 == num:
        return env.actions.pickup
    elif 4 == num:
        return env.actions.drop
    else:
        return env.actions.toggle

def main():
    global env

    # setup
    env = gym.make('MiniGrid-SimpleCrossingS9N1-v0')
    env = FullyObsWrapper(env)
    env.seed(1)
    s = env.reset()
    s = s['image']
    curr_dir = tuple(env.dir_vec)
    #agent = DynaQ_Agent(0.95, 0.01, 1.0, 0.99999, 30)
    agent = DynaQ_Agent(0.95, 0.01, 0.5, 0.9999, 30)
    if load_model:
        agent.load("./grid_save/grid_dynaQ.h5")

    # some vars
    r_sum = 0
    total_r = None
    preproc = False

    run = 0
    step_num = 0
    while True:
        step_num += 1

        if visualize:
            env.render()

        # preprocess
        if not preproc:
            s = preprocess(s)
        preproc = False

        # get action
        a,_ = agent.act(s)
        a_code = get_action(a)

        s_prime, r, done, _ = env.step(a_code)
        s_prime = s_prime['image']
        if not test_model:
            r_sum += r
            s_prime = preprocess(s_prime)
            preproc = True
            s = s_prime

            curr_dir = tuple(env.dir_vec)
            agent.update(s, a, r, s_prime, done, curr_dir)

            if 0 == step_num % 10:
                print("Agent check: took action {}, current r_sum = {}".format(a, r_sum))

            if done:
                run += 1

                if 0 == run % 20:
                    agent.save("./grid_save/grid_dynaQ.h5")

                #if 0 == r_sum:
                #    r_sum = -1

                total_r = r_sum if total_r is None else total_r * 0.95 + r_sum * 0.05 # scaled off of gamma
                print("At run #{}".format(run))
                print("resetting env! episode reward total was {} w/ running mean {}".format(r_sum, total_r))
                print("current exploration constant: {:.2}".format(agent.epsilon))
                print()

                r_sum = 0
                step_num = 0
                env.seed(1)
                s = env.reset()
                s = s['image']
                preproc = False
        else:
            r_sum += r
            s = s_prime

            if done:
                print("finished w/ score {}!".format(r_sum))

                r_sum = 0
                step_num = 0
                env.seed(1)
                s = env.reset()
                s = s['image']

if __name__ == "__main__":
    main()
