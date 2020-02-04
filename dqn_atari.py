# NOTE as this is primarily a try-it-out program, many of the techniques in here (mostly related to visualization) are NOT COMPLETE/OPTIMAL, make sure to fix/adapt before uncommenting or copying to new programs

import math
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import gym
import datetime
# for ref: https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
# note: o/w try out pyqtgraph
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

debug = False

r_sum = 0
batch_r_sum = 0
run = 0
iteration_val = 0
iteration = []
epsilon_log = []
exploit_phase = 0

load_model = True
visualize = True
test_model = True
random_agent = False

# init data-tracking stuff
#log_dir =".\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#fig = plt.figure(1)
#ax = fig.add_subplot(111)
#ax.set_title("Exploration Tracker")
#ax.set_xlabel("run")
#ax.set_ylabel("exploration probability")
#l, = ax.plot(iteration, epsilon_log, 'g-')

def update(frame):
    global fig

    fig = plt.figure(1)
    l.set_data(iteration, epsilon_log)
    fig.gca().relim()
    fig.gca().autoscale_view()

    return l,

#anim = FuncAnimation(fig, update, interval=1000, blit=True) # faster
# -OR-
#anim = FuncAnimation(fig, update, interval=1000) # plot updates correctly
#plt.show(block=False)

# NOTE: this version of the DQN does not explicitly perform importance sampling or really regard any interpretation of the behavior-target policies BECAUSE DQN samples over *state-transitions* NOT policies

class DQNAgent:
    def __init__(self, s_size, a_size):
        self.s_size = s_size
        self.a_size = a_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        if not load_model:
            self.epsilon = 1.0
        else:
            self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self.build()

    def build(self):
        model = models.Sequential()

        # Convolutional layers, want to be big enough to grasp relevant regions (i.e. boxers)
        model.add(layers.Conv2D(3, (16,16), activation='tanh', padding='same', input_shape=(80,75,3))) # for multi-channel (e.g. Boxing)
        #model.add(layers.Conv2D(3, (16,16), activation='tanh', padding='same', input_shape=(80,80,3))) # for else (e.g. Pong)
        model.add(layers.Conv2D(3, (8,8), activation='tanh', padding='same'))
        model.add(layers.Conv2D(3, (4,4), activation='tanh', padding='same'))
        model.add(layers.Conv2D(3, (2,2), activation='tanh', padding='same'))

        model.add(layers.Flatten())

        model.add(layers.Dense(512, activation='tanh'))
        model.add(layers.Dense(512, activation='tanh'))
        model.add(layers.Dense(self.a_size, activation='linear'))

        # DEBUG
        if debug:
            model.summary()
            input("wait")

        model.compile(loss='mse', optimizer='adam')

        return model

    def remember(self, s, a, r, s_prime, done):
        self.memory.append((s, a, r, s_prime, done))

    def act(self, s):
        if not test_model:
            if np.random.rand() <= self.epsilon:

                # DEBUG
                if debug: print("exploring!")

                return random.randrange(self.a_size)

        if not random_agent:
            q_vals = self.model.predict(np.asarray([s], dtype=np.float32))
        else:
            return random.randrange(self.a_size)
        return np.argmax(q_vals[0])

    def replay(self, batch_size):
        global batch_r_sum
        global iteration_val
        global iteration
        global epsilon_log
        global exploit_phase
        global log_dir
        global fig
        global ax

        #tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        batch = random.sample(self.memory, batch_size)
        s_in = []
        a_out = []
        for s, a, r, s_prime, done in batch:
            target = r
            if not done:
                target = (r + self.gamma * np.amax(self.model.predict(np.asarray([s_prime], dtype=np.float32))[0]))
            target_f = self.model.predict(np.asarray([s], dtype=np.float32))
            target_f[0][a] = target
            s_in.append(s)
            a_out.append(target_f[0])
        self.model.fit(np.asarray(s_in), np.asarray(a_out), verbose=0)
        #self.model.fit(np.asarray(s_in), np.asarray(a_out), verbose=0, callbacks=[tb_callback]) # TODO maybe some way to amplify the learning of good performances? but maybe through learning it somehow (either inherenty or through another model)?
        if not test_model: # custom exploration modulation fxn
            # DEBUG
            #if False:
            if not load_model and run < 20: # force some initial exploration and decay
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)

                # plotting
                #iteration.append(iteration_val)
                #iteration_val += 1
                #epsilon_log.append(self.epsilon)

                print("still in forced exploration, next epsilon: {}".format(self.epsilon))
            else: # then use modulation
                if exploit_phase < 90:
                    #old_epsilon = self.epsilon
                    ## modulation happens over the r_sum of the batch NOT total for the episode
                    #self.epsilon = 1.0 / (1 + math.exp((1/5)*(batch_r_sum-10)))

                    ## smoothing
                    #if self.epsilon <= old_epsilon:
                    #    self.epsilon = max(self.epsilon, old_epsilon - 0.1)
                    #else:
                    #    self.epsilon = min(self.epsilon, old_epsilon + 0.1)

                    print("in modulated exploration, current batch_r_sum: {}, next (smoothed) epsilon: {}".format(batch_r_sum, self.epsilon))

                    exploit_phase += 1
                    if 90 == exploit_phase:
                        batch_r_sum = 0
                else: # we sometimes force exploitation which is a metric for overall progress
                    self.epsilon = self.epsilon_min
                    if exploit_phase < 110:
                        exploit_phase += 1
                    else:
                        self.epsilon = 1.0 / (1 + math.exp((1/5)*(batch_r_sum-10)))
                        exploit_phase = 0
                        batch_r_sum = 0

                    print("in EXPLOITATION, current exploit batch_r_sum: {}, epsilon (for ref): {}".format(batch_r_sum, self.epsilon))

                #iteration.append(iteration_val)
                #iteration_val += 1
                #epsilon_log.append(self.epsilon)

    def load(self, f):
        print("loading model!")
        self.model.load_weights(f)

    def save(self, f):
        print("saving model!")
        self.model.save_weights(f)

# NOTE: sparse matrices (like these images) are not inherently bad, models can generally handle them BUT the problem arises w/ the fact that the matrices are now largely uninformative and waste space
# for Boxing: preprocess image frame into 80 x 75 float vector w/ standardized background/object colors
def preprocess_Boxing(I):
    I = I[:,10:]
    I = I[25:185]
    I = I[::2,::2,:]

    I[I == 0] = 100 # opponent color

    I[I == 110] = 0
    I[I == 195] = 0
    I[I == 156] = 0
    I[I == 66] = 0
    I[I == 144] = 0
    I[I == 61] = 0

    I[I > 100] = 200 # player color

    # DEBUG
    if debug:
        print("final image:")
        #print(I)
        #print(I.shape)
        fig = plt.figure(2)
        plt.imshow(I, interpolation="nearest")
        plt.show()

    #return I.astype(np.float).ravel()
    return I

# for Pong: same story but w/ dims 80 x 80
def preprocess_Pong(I):
    I = I[35:195]
    I = I[::2,::2,:]

    I[I == 144] = 0
    I[I == 109] = 0
    I[I == 72] = 0
    I[I == 17] = 0
    I[I == 186] = 100
    I[I == 92] = 100
    I[I == 213] = 200
    I[I == 130] = 200
    I[I == 74] = 200

    # DEBUG
    if debug:
        print("final image:")
        #print(I)
        #print(I.shape)
        fig = plt.figure(2)
        plt.imshow(I, interpolation="nearest")
        plt.show()

    #return I.astype(np.float).ravel()
    return I

def main():
    global run
    global r_sum
    global batch_r_sum

    env = gym.make("Boxing-v0")
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    agent = DQNAgent(s_size, a_size)
    if load_model:
        agent.load("./atari_save/boxing_dqn.h5")

    # some vars
    preproc = False
    batch_size = 64
    mem_stored = 0
    total_r = None
    s = env.reset()

    while True:
        if visualize:
            env.render()

        # preprocess
        if not preproc:
            s = preprocess_Boxing(s)
        preproc = False

        # get action
        a = agent.act(s)

        # DEBUG
        if debug:
            #for layer in agent.model.layers: print(layer.get_config(), layer.get_weights())
            print("action: {}".format(a))
            # test visualize internal layers on image :)
            conv_layer_outputs = [layer.output for layer in agent.model.layers[:3]]
            activations_extractor = models.Model(inputs=agent.model.input, outputs=conv_layer_outputs)
            activations = activations_extractor.predict(np.asarray([s]))
            fig = plt.figure(3)
            k = 1
            for j,act in enumerate(activations):
                num_feat = act.shape[-1]
                for i in range(num_feat):
                    channel_img = act[0,:,:,i]
                    fig.add_subplot(3,3,k)
                    k += 1
                    plt.imshow(channel_img, interpolation="nearest", aspect="auto", cmap="viridis")
            input("wait")

        s_prime, r, done, _ = env.step(a)
        if not test_model:
            r = r if not done else -r
            r_sum += r
            batch_r_sum += r
            s_prime = preprocess_Boxing(s_prime)
            preproc = True
            agent.remember(s, a, r, s_prime, done)
            mem_stored += 1
            s = s_prime

            if mem_stored > batch_size:
                agent.replay(batch_size)
                mem_stored = 0

            if done:
                run += 1

                if 0 == run % 10:
                    agent.save("./atari_save/boxing_dqn.h5")

                total_r = r_sum if total_r is None else total_r * 0.95 + r_sum * 0.05 # scaled off of gamma
                print("At run #{}".format(run))
                print("resetting env! episode reward total was {} w/ running mean {}".format(r_sum, total_r))
                print("current exploration constant: {:.2}".format(agent.epsilon))
                print()

                r_sum = 0
                s = env.reset()
                preproc = False
        else:
            r = r if not done else -r
            r_sum += r
            s = s_prime

            if done:
                print("ended with {} points!".format(r_sum))

                r_sum = 0
                s = env.reset()

if __name__ == "__main__":
    main()
