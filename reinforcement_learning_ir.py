# -*- coding: utf-8 -*-
"""Reinforcement Learning - IR.ipynb

# Algoritmo de Reinforcement Learning(off-policy com q-learning)
## Trabalho final de Introdução à Robótica: Gabriel Sousa Silva Costa
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

"""### Inicializando constantes"""

MAP_SIZE = (9, 14)
EPOCHS = 500
STEPS_PER_EPOCH = 1000
MOVE_PENALTY = 30
# NOT_MOVE_PENALTY = 10
OBSTACLE_PENALTY = 600
FINISH_REWARD = 600

epsilon = 0
EPS_DECAY = 0.9998
SHOW_EVERY = 1

start_q_table = "qtable-1608060269.pickle"
lr = 0.1
discount = 0.95

AGENT_ID = 1
OBJECTIVE_ID = 2
ENEMY_ID = 3

d = {1: (255, 50, 0),
     2: (0, 255, 0),
     3: (0, 0, 225)}

spawn_point = (3, 1)
destiny_point = (3, 11)
upper_bound = (8, 13)
lower_bound = (0, 0)

# Definição das ações(vetor de movimento)
movement_vector = np.array(
    [
    [0, 0],
    [0, 1],
    [0, -1],
    [1, 0],
    [1, 1],
    [1, -1],
    [-1, 0],
    [-1, 1],
    [-1, -1]                        
    ]
)

"""### Inicializando classes"""

class Entity:
    def __init__(self, entity_id, enemy_start = 7):
        if entity_id == AGENT_ID:
            self.x, self.y = spawn_point
        elif entity_id == OBJECTIVE_ID:
            self.x, self.y = destiny_point
        else:
            self.x, self.y = (np.random.randint(0, MAP_SIZE[0]), enemy_start)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def getPosition(self):
        return (self.x, self.y)

    def action(self, entity_id, choice=None):
        if entity_id == AGENT_ID:
            self.move(AGENT_ID, movement_vector[choice][0], movement_vector[choice][1])
        elif entity_id == ENEMY_ID:
            self.move(ENEMY_ID, np.random.randint(1, 4), 0)

    def move(self, entity_id, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
            if self.x > upper_bound[0]:
                self.x = self.x % MAP_SIZE[0]
        if not y and entity_id != ENEMY_ID:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > upper_bound[0]:
            self.x = upper_bound[0]
        if self.y < 0:
            self.y = 0
        elif self.y > upper_bound[1]:
            self.y = upper_bound[1]

if start_q_table is None:
    print("Loading new")
    q_table = np.full([MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[0], movement_vector.shape[0]], 0)
else:
    print("Loading from saved")
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

epoch_rewards = []

for epoch in range(EPOCHS):
    agent = Entity(AGENT_ID)
    objective = Entity(OBJECTIVE_ID)
    enemy = Entity(ENEMY_ID)
    if epoch % SHOW_EVERY == 0:
        print(f"on # {epoch}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} episode mean {np.mean(epoch_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    epoch_reward = 0
    for i in range(STEPS_PER_EPOCH):
        observation = (agent.getPosition(), enemy.getPosition())
        if np.random.random() > epsilon:
            action = np.argmax(q_table[observation[0][0], observation[0][1], observation[1][0]])
        else:
            action = np.random.randint(0, movement_vector.shape[0])

        # prev_agent_position = agent.getPosition()
        agent.action(AGENT_ID, action)
        enemy.action(ENEMY_ID)

        if agent.x == enemy.x and agent.y == enemy.y:
           reward = - OBSTACLE_PENALTY
        elif agent.x == objective.x and agent.y == objective.y:
            reward = FINISH_REWARD
        # elif prev_agent_position[0] == agent.x and prev_agent_position[1] == agent.y:
        #     reward = - NOT_MOVE_PENALTY
        else:
            reward = - MOVE_PENALTY

        new_observation = (agent.getPosition(), enemy.getPosition())
        max_future_q = np.max(q_table[new_observation[0][0], new_observation[0][1], new_observation[1][0]])
        current_q = q_table[observation[0][0], observation[0][1], observation[1][0], action]

        if reward == FINISH_REWARD:
            new_q = FINISH_REWARD
        elif reward == -OBSTACLE_PENALTY:
            new_q = -OBSTACLE_PENALTY
        else:
            new_q = (1-lr)*current_q + lr*(reward + discount*max_future_q)

        q_table[observation[0][0], observation[0][1], observation[1][0], action] = new_q

        if show:
            environment = np.zeros((MAP_SIZE[0], MAP_SIZE[1], 3), dtype = np.uint8)
            environment[objective.x, objective.y] = d[OBJECTIVE_ID]
            environment[agent.x, agent.y] = d[AGENT_ID]
            environment[enemy.x, enemy.y] = d[ENEMY_ID]

            img = Image.fromarray(environment, "RGB")
            img = img.resize((300, 300))
            cv2.imshow("", np.array(img))
            if reward==FINISH_REWARD or reward == -OBSTACLE_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break

        epoch_reward += reward
        if reward==FINISH_REWARD or reward == -OBSTACLE_PENALTY:
            break
    epoch_rewards.append(epoch_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(epoch_rewards, np.ones((SHOW_EVERY, ))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)