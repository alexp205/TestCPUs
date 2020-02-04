import numpy as np
import pickle
import gym

# hyperparams
H = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99

resume = False
render = True
test_model = False

D = 80 * 80
if resume:
    model = pickle.load(open("model.p", "rb"))
else:
    model = {}
    model["W1"] = np.random.randn(H, D) / np.sqrt(D)
    model["W2"] = np.random.randn(H) / np.sqrt(H)

grad_arr = {k:np.zeros_like(v) for k,v in model.items()}
rmsprop_cache = {k:np.zeros_like(v) for k,v in model.items()}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# preprocess image frame into 80 x 80 float vector w/ standardized background/object colors
def preprocess(I):
    I = I[35:195]
    I = I[::2,::2,0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1

    return I.astype(np.float).ravel()

# take rewards over time and compute discounted reward
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    sum_term = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: sum_term = 0
        sum_term = sum_term * gamma + r[t]
        discounted_r[t] = sum_term
    
    return discounted_r

def policy_forward(x):
    h = np.dot(model["W1"], x)
    h[h<0] = 0
    logp = np.dot(model["W2"], h)
    p = sigmoid(logp)

    return p,h

# backward pass (for updates)
def policy_backward(eph, epdlogp):
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model["W2"])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)

    return {"W1":dW1, "W2":dW2}

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
xs, hs, dlogps, drs = [], [], [], []
total_reward = None
reward_sum = 0
episode_num = 0

while True:
    if render: env.render()

    # preprocess observation from game
    curr_x = preprocess(observation)
    x = curr_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = curr_x

    # run screen through network and get action from probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3

    if not test_model:
        # record info needed for backpropagation
        xs.append(x)
        hs.append(h)
        y = 1 if action == 2 else 0
        dlogps.append(y - aprob)

        # step env and get new msmt
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward)

        if done:
            episode_num += 1

            # stack all info for episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], []

            # compute discounted rewards, but backwards!
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr
            grad = policy_backward(eph, epdlogp)
            for k in model: grad_arr[k] += grad[k]

            # do rmsprop (backpropagation) updates
            if episode_num % batch_size == 0:
                for k,v in model.items():
                    g = grad_arr[k]
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_arr[k] = np.zeros_like(v)

            # necessary stuff
            total_reward = reward_sum if total_reward is None else total_reward * 0.99 + reward_sum * 0.01
            print("resetting env, episode reward total was {} w/ running mean {}".format(reward_sum, total_reward))
            if episode_num % 100 == 0: pickle.dump(model, open("model.p", "wb"))
            reward_sum = 0
            observation = env.reset()
            prev_x = None

        if reward != 0:
            print("ep {}: game finished, reward: {}".format(episode_num, reward))
            if reward == -1:
                print("")
            else:
                print(" !!!!!!!!")
    else:
        # step env and get new msmt
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
        if reward != 0:
            print("game finished, reward: {}".format(episode_num, reward))
