from random import random, randrange
import pickle
import gym_minigrid
import gym
from gym_minigrid.wrappers import *

debug = False

load_model = False
visualize = True
test_model = False
env = None

# the SARSA(lambda) agent class
class SARSA_lambda_Agent:
    def __init__(self, gamma, alpha, _lambda, epsilon, epsilon_decay, curr_dir):
        self.gamma = gamma
        self.alpha = alpha
        self._lambda = _lambda
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.elig_traces = np.zeros(81)
        self.weights = np.zeros(81)
        self.curr_dir = curr_dir
        self.explore_mgr = 0

    """
    # Map of agent direction indices to vectors
    DIR_TO_VEC = [
        # Pointing right (positive X)
        np.array((1, 0)),
        # Down (positive Y)
        np.array((0, 1)),
        # Pointing left (negative X)
        np.array((-1, 0)),
        # Up (negative Y)
        np.array((0, -1)),
    ]
    """
    def act(self, s):
        env_width = 9
        env_height = 9
        cand_a = None
        cand_Q = None
        # get greedy action
        if random() < self.epsilon:

            # DEBUG
            if debug: print("in explore")

            cand_a = randrange(6)
            cand_Q = 0
            if 2 == cand_a:
                s_step = s
                for i,f in enumerate(s):
                    if 10 == f:
                        if (1,0) == self.curr_dir:
                            if i+env_width < 81:
                                next_val = s_step[i+env_width]
                                if next_val != 0:
                                    s_step[i] = 0
                                    s_step[i+env_width] = 10
                        elif (0,1) == self.curr_dir:
                            if 8 != i%9:
                                next_val = s_step[i+1]
                                if next_val != 0:
                                    s_step[i] = 0
                                    s_step[i+1] = 10
                        elif (-1,0) == self.curr_dir:
                            if i-env_width >= 0:
                                next_val = s_step[i-env_width]
                                if next_val != 0:
                                    s_step[i] = 0
                                    s_step[i-env_width] = 10
                        else:
                            if 0 != i%9:
                                next_val = s_step[i-1]
                                if next_val != 0:
                                    s_step[i] = 0
                                    s_step[i-1] = 10
                        break
                for i,f in enumerate(s_step):
                    cand_Q += f * self.weights[i]
            else:
                for i,f in enumerate(s):
                    cand_Q += f * self.weights[i]
        else:

            # DEBUG
            if debug: print("in exploit")

            other_Q = 0
            for i,f in enumerate(s):
                other_Q += f * self.weights[i]
            s_step = s
            for i,f in enumerate(s):
                if 10 == f:
                    if (1,0) == self.curr_dir:
                         if i+env_width < 81:
                             next_val = s_step[i+env_width]
                             if next_val != 0:
                                 s_step[i] = 0
                                 s_step[i+env_width] = 10
                    elif (0,1) == self.curr_dir:
                        if 8 != i%9:
                             next_val = s_step[i+1]
                             if next_val != 0:
                                 s_step[i] = 0
                                 s_step[i+1] = 10
                    elif (-1,0) == self.curr_dir:
                         if i-env_width >= 0:
                             next_val = s_step[i-env_width]
                             if next_val != 0:
                                 s_step[i] = 0
                                 s_step[i-env_width] = 10
                    else:
                         if 0 != i%9:
                             next_val = s_step[i-1]
                             if next_val != 0:
                                 s_step[i] = 0
                                 s_step[i-1] = 10
                    break
            step_Q = 0
            for i,f in enumerate(s_step):
                step_Q += f * self.weights[i]
            max_val = max(other_Q, step_Q)
            if step_Q == other_Q:
                cand_a = randrange(6)
                cand_Q = step_Q
            elif max_val == step_Q:
                cand_a = 2
                cand_Q = step_Q
            else:
                cand_a = randrange(6)
                while 2 != cand_a:
                    cand_a = randrange(6)
                cand_Q = other_Q

        return cand_a, cand_Q

    def update(self, s, a, r, s_prime, done, curr_dir):
        self.explore_mgr += 1

        self.curr_dir = curr_dir
        net_w = 0
        for i,f in enumerate(s):
            net_w += f * self.weights[i]
            if 10 == f:
                # TODO
                self.elig_traces[i] = 1 # replacing traces

        error = r - net_w
        
        if done:
            self.weights += self.alpha * error * self.elig_traces
        else:
            a_prime, Q_prime_a = self.act(s_prime)
            error = error + self.gamma * Q_prime_a
            self.weights += self.alpha * error * self.elig_traces
            self.elig_traces = self.gamma * self._lambda * self.elig_traces

        # DEBUG
        #if debug:
        if True:
            print("Weight vis")
            for i in range(9):
                print(self.weights[i*9:((i+1)*9)])
            print("E-trace vis")
            for i in range(9):
                print(self.elig_traces[i*9:((i+1)*9)])
            input("wait")

        if not test_model and 0 == self.explore_mgr%100:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            self.explore_mgr = 0
        if 0 == self.explore_mgr%1000:
            print("current exploration constant: {}".format(self.epsilon))

    def reset_elig_traces(self):
        self.elig_traces = np.zeros(81)

    def load(self, f):
        print("loading model!")
        self.weights = pickle.load(open(f, "rb"))

    def save(self, f):
        print("saving model!")
        pickle.dump(self.weights, open(f, "wb"))

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
    agent = SARSA_lambda_Agent(0.95, 0.01, 0.9, 1.0, 0.999, curr_dir)
    if load_model:
        agent.load("./grid_save/grid_sarsa-lambda.h5")

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

                if 0 == run % 50:
                    agent.save("./grid_save/grid_sarsa-lambda.h5")

                #if 0 == r_sum:
                #    r_sum = -1

                total_r = r_sum if total_r is None else total_r * 0.95 + r_sum * 0.05 # scaled off of gamma
                print("At run #{}".format(run))
                print("resetting env! episode reward total was {} w/ running mean {}".format(r_sum, total_r))
                print("current exploration constant: {:.2}".format(agent.epsilon))
                print()

                r_sum = 0
                step_num = 0
                agent.reset_elig_traces()
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
