# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:40:30 2020

@author: agabh
"""

import math
import random
import pygame
import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.stats import bernoulli
import pandas as pd

class cube(object):
    
    def __init__(self,start,width, rows,dirnx=1,dirny=0,color=(255,0,0)):
        self.pos = start
        self.dirnx = 1
        self.dirny = 0
        self.color = color
        self.w = width
        self.rows = rows
        
        pos_x = self.pos[0]
        pos_y = self.pos[1]
        if pos_x >= self.rows:
            pos_x = pos_x - self.rows
        if pos_y >= self.rows:
            pos_y = pos_y - self.rows
        if pos_x < 0:
            pos_x += self.rows
        if pos_y < 0:
            pos_y += self.rows
        self.pos = (pos_x, pos_y)
 
       
    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        pos_x = self.pos[0] + self.dirnx
        pos_y = self.pos[1] + self.dirny
        if pos_x >= self.rows:
            pos_x = pos_x - self.rows
        if pos_y >= self.rows:
            pos_y = pos_y - self.rows
        if pos_x < 0:
            pos_x += self.rows
        if pos_y < 0:
            pos_y += self.rows
        self.pos = (pos_x, pos_y)
 
    def draw(self, surface, eyes=False):
        dis = self.w // self.rows
        i = self.pos[0]
        j = self.pos[1]
 
        pygame.draw.rect(surface, self.color, (i*dis+1,j*dis+1, dis-2, dis-2))
        if eyes:
            centre = dis//2
            radius = 3
            circleMiddle = (i*dis+centre-radius,j*dis+8)
            circleMiddle2 = (i*dis + dis -radius*2, j*dis+8)
            pygame.draw.circle(surface, (0,0,0), circleMiddle, radius)
            pygame.draw.circle(surface, (0,0,0), circleMiddle2, radius)
       
 
class snake(object):
    body = []
    turns = {}
    def __init__(self, color, pos, width, rows, last_move):
        self.color = color
        self.head = cube(pos, width, rows)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.last_move = last_move
        self.width = width
        self.rows = rows
 
    def move(self, key):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        #key = random_move(len(self.body), self.last_move)
        self.last_move = key
 
        if key == -1 :
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
 
        elif key == 1:
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
 
        elif key == 10:
            self.dirnx = 0
            self.dirny = -1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
 
        elif key == -10:
            self.dirnx = 0
            self.dirny = 1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

 
        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0],turn[1])
                if i == len(self.body)-1:
                    self.turns.pop(p)
            else:
                if c.dirnx == -1 and c.pos[0] <= 0: 
                    c.pos = (c.rows + c.pos[0] - 1, c.pos[1])
                elif c.dirnx == 1 and c.pos[0] >= c.rows - 1: 
                    c.pos = (c.pos[0] - c.rows + 1, c.pos[1])
                elif c.dirny == 1 and c.pos[1] >= c.rows - 1: 
                    c.pos = (c.pos[0], c.pos[1] - c.rows + 1)
                elif c.dirny == -1 and c.pos[1] <= 0: 
                    c.pos = (c.pos[0], c.rows + c.pos[1] - 1)
                else: 
                    c.move(c.dirnx,c.dirny)
       
 
    def reset(self, pos):
        self.head = cube(pos, self.width, self.rows)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1
 
 
    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            if tail.pos[0] - 1 > 0 :
                self.body.append(cube((tail.pos[0]-1,tail.pos[1]), self.width, self.rows))
            else : 
                self.body.append(cube((tail.pos[0] - 1 + self.rows ,tail.pos[1]), self.width, self.rows))
        elif dx == -1 and dy == 0:
            if tail.pos[0] + 1 < self.rows :
                self.body.append(cube((tail.pos[0]+1,tail.pos[1]), self.width, self.rows))
            else :
                self.body.append(cube((tail.pos[0] + 1 - self.rows, tail.pos[1]), self.width, self.rows))
        elif dx == 0 and dy == 1:
            if tail.pos[1] - 1 > 0 :
                self.body.append(cube((tail.pos[0], tail.pos[1] - 1), self.width, self.rows))
            else :
                self.body.append(cube((tail.pos[0], tail.pos[1] - 1 + self.rows), self.width, self.rows))
        elif dx == 0 and dy == -1:
            if tail.pos[1] + 1 < self.rows : 
                self.body.append(cube((tail.pos[0], tail.pos[1] + 1), self.width, self.rows))
            else :
                self.body.append(cube((tail.pos[0], tail.pos[1] + 1 - self.rows), self.width, self.rows))
 
        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy
       
 
    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i ==0:
                c.draw(surface, True)
            else:
                c.draw(surface)
 
 
def drawGrid(w, rows, surface):
    sizeBtwn = w // rows
 
    x = 0
    y = 0
    for l in range(rows):
        x = x + sizeBtwn
        y = y + sizeBtwn
 
        pygame.draw.line(surface, (255,255,255), (x,0),(x,w))
        pygame.draw.line(surface, (255,255,255), (0,y),(w,y))
       
 
def redrawWindow(surface):
    global rows, width, s, snack
    surface.fill((0,0,0))
    s.draw(surface)
    snack.draw(surface)
    drawGrid(width,rows, surface)
    pygame.display.update()
 
 
def randomSnack(rows, item):
 
    positions = item.body
 
    while True:
        x = random.randrange(rows)
        y = random.randrange(rows)
        if len(list(filter(lambda z:z.pos == (x,y), positions))) > 0:
            continue
        else:
            break
       
    return (x,y)

# in case if the snake's length is >1, there will be one less legal action to 
# choose from, depending on the current direction or last action
def random_move(snake_len, last_move):
    actions = [1, -1, 10, -10]
    if snake_len < 2:
        move = random.choice(actions)
    else:
        actions2 = [x for x in actions if x != -1 * last_move]
        move = random.choice(actions2)
    return move

global eps, state_counter, q_mat
eps = 0.2
alpha = 0.7
gamma = 0.9

#state to be defined as the direction in which the snake was moving as the result 
# of previous action,and the corresponding horizontal and vertical distance b/w current 
# pos and the snack
#update_1 - the state should also include the info about the snake's body to avoid
# the snake from hitting itself

# the total no of states will be = n_rows * n_columns * 4(no. of possible directions 
# resulting from previous action)

# states = np.zeros((rows, rows, 4))

def snake_mat(rows,snake):
    mat = np.zeros((rows, rows))
    for cube in snake.body:
        mat[cube.pos[0]][cube.pos[1]] = 1
    return mat

#objective - to determine a kXk matrix around the snake's head to scan for body parts
#function to return the x & y indices to scan for
# def snake_surround(snake, k, rows):
#     x = snake.body[0].pos[0]
#     y = snake.body[0].pos[1]
#     x_indices = np.arange(x - k, x + k + 1)
#     y_indices = np.arange(y - k, y + k + 1)
    
#     x_2 = snake.body[1].pos[0]
#     y_2 = snake.body[1].pos[0]
#     dir_x = x - x_2
#     dir_y = y - y_2
    
#     for x_i in x_indices:
#         if x_i < 0:
#             x_i += rows
#         elif x_i >= rows:
#             x_i -= rows
#     for y_i in y_indices:
#         if y_i < 0:
#             y_i += rows
#         elif y_i >= rows:
#             y_i -= rows
            
#     x_indices = list(x_indices)
#     y_indices = list(y_indices)
    
#     if abs(dir_x) > 0:
#         x_indices.remove(x)
#     if abs(dir_y) > 0:
#         y_indices.remove(y)
#     return x_indices, y_indices
    
#the objective is to evaluate the snake's head near surroundings to check for
#self body blocks. this is broken into 3 parts - in the direction of snake's
#movement, left and right to it.
    
# to achieve this 3 functions are made which will evaluate the nearby body parts
    

def snake_gap_up(x_pos, y_pos, x_dir, y_dir, k, snake_mat):
    if x_dir < 0:
        indices = np.arange(x_pos - k, x_pos)
    elif x_dir > 0:
        indices = np.arange(x_pos + 1, x_pos + k + 1)
    elif y_dir < 0:
        indices = np.arange(y_pos - k, y_pos)
    elif y_dir > 0:   
        indices = np.arange(y_pos + 1, y_pos + k + 1)
    
    for i in range(0, len(indices)):
        if indices[i] < 0:
            indices[i] += rows
        elif indices[i] >= rows:
            indices[i] -= rows

    gap_up = 0
    if abs(x_dir) > 0 :
        for i in indices:
            if snake_mat[i][y_pos] > 0 and gap_up < 1:
                gap_up += 1
            else:
                break
    elif abs(y_dir) > 0:
        for i in indices:
            if snake_mat[x_pos][i] > 0 and gap_up < 1:
                gap_up += 1
            else:
                break
    return gap_up
    
def snake_gap_left(x_pos, y_pos, x_dir, y_dir, k, snake_mat):
    if x_dir < 0:
        indices = np.arange(y_pos + 1, y_pos + k + 1)
    elif x_dir > 0:
        indices = np.arange(y_pos - k, y_pos)
    elif y_dir < 0:
        indices = np.arange(x_pos - k, x_pos)
    elif y_dir > 0:   
        indices = np.arange(x_pos + 1, x_pos + k + 1)
       
    for i in range(0, len(indices)):
        if indices[i] < 0:
            indices[i] += rows
        elif indices[i] >= rows:
            indices[i] -= rows
    
    gap_left = 0
    if abs(x_dir) > 0 :
        for i in indices:
            if snake_mat[x_pos][i] > 0 and gap_left < 1:
                gap_left += 1
            else:
                break
    elif abs(y_dir) > 0:
        for i in indices:
            if snake_mat[i][y_pos] > 0 and gap_left < 1:
                gap_left += 1
            else:
                break
    return gap_left

def snake_gap_right(x_pos, y_pos, x_dir, y_dir, k, snake_mat):
    if x_dir < 0:
        indices = np.arange(y_pos - k, y_pos)
    elif x_dir > 0:
        indices = np.arange(y_pos + 1, y_pos + k + 1)
    elif y_dir < 0:
        indices = np.arange(x_pos + 1, x_pos + k + 1)
    elif y_dir > 0:   
        indices = np.arange(x_pos - k, x_pos)
       
    for i in range(0, len(indices)):
        if indices[i] < 0:
            indices[i] += rows
        elif indices[i] >= rows:
            indices[i] -= rows
    
    gap_right = 0
    if abs(x_dir) > 0 :
        for i in indices:
            if snake_mat[x_pos][i] > 0 and gap_right < 1:
                gap_right += 1
            else:
                break
    elif abs(y_dir) > 0:
        for i in indices:
            if snake_mat[i][y_pos] > 0 and gap_right < 1:
                gap_right += 1
            else:
                break
    return gap_right

def current_state(prev_action, snake, snack, rows, k):
    snake_head_pos = snake.body[0].pos
    snack_pos = snack.pos
    x_pos = snake_head_pos[0]
    y_pos = snake_head_pos[1]
    dis_x = np.abs(snake_head_pos[0] - snack_pos[0])
    dis_y = np.abs(snake_head_pos[1] - snack_pos[1])
    snake_len = len(snake.body)
    snake_len_f = int(snake_len > 1)
    gap_up = 0
    gap_left = 0
    gap_right = 0
    
    if len(snake.body) > 2:
        dir_x = x_pos - snake.body[1].pos[0]
        dir_y = y_pos - snake.body[1].pos[1]
        
        snake_pos_mat = snake_mat(rows, snake)
        
        gap_up = snake_gap_up(x_pos, y_pos, dir_x, dir_y, k, snake_pos_mat)
        gap_left = snake_gap_left(x_pos, y_pos, dir_x, dir_y, k, snake_pos_mat)
        gap_right = snake_gap_right(x_pos, y_pos, dir_x, dir_y, k, snake_pos_mat)
        
    return prev_action, dis_x, dis_y, gap_up, gap_left, gap_right, snake_len_f


global state_dict, prev_actions, width, rows

state_counter = 0
state_dict = {}
rows = 10
width = 200

prev_actions = [1, -1, 10, -10]
dis_xs = np.arange(0, rows)
dis_ys = np.arange(0, rows)
gaps_up = [0, 1]
gaps_left = [0, 1]
gaps_right = [0, 1]
snake_len_v = [0, 1]

for a in prev_actions:
    for b in dis_xs:
        for c in dis_ys:
            for d in gaps_up:
                for e in gaps_left:
                    for f in gaps_right:
                        for g in snake_len_v:
                            state_dict[a,b,c,d,e,f,g] = state_counter
                            state_counter += 1
                            
global q_mat
q_mat = np.zeros((len(state_dict), len(prev_actions)))
                        
def q_matrix_update(current_state, current_action, next_state, snack, dead):
    state_counter = state_dict[current_state]
    state_counter_next = state_dict[next_state]
    current_action_index = prev_actions.index(current_action)
    reward = reward_function(current_state, next_state, snack, dead)
    max_reward_next_state = np.max(q_mat[state_counter_next])
    delta = alpha * (reward + gamma * max_reward_next_state - q_mat[state_counter, current_action_index])
    q_mat[state_counter, current_action_index] += delta
    return q_mat

def reward_function(current_state, next_state, snack, dead):
    old_dis_x = current_state[1]
    old_dis_y = current_state[2]
    new_dis_x = next_state[1]
    new_dis_y = next_state[2]
    
    gap_up_current = current_state[3]
    gap_left_current = current_state[4]
    gap_righ_current = current_state[5]
    
    gap_up_next = next_state[3]
    gap_left_next = next_state[4]
    gap_right_next = next_state[5]
    
    old_dis = np.sqrt((old_dis_x**2 + old_dis_y**2))
    new_dis = np.sqrt((new_dis_x**2 + new_dis_y**2))
    
    if old_dis < new_dis :
        reward_dis = -100 * (new_dis - old_dis)
    else :
        reward_dis = 100 * (old_dis - new_dis)
    
    if snack == 1:
        reward_snack = 1000
    else :
        reward_snack = 0
        
    if dead == 1 :
        reward_dead = -10000
    else :
        reward_dead = 0
        
    if gap_up_next + gap_left_next + gap_right_next == 2:
        reward_gap_next = -100
    elif gap_up_next + gap_left_next + gap_right_next == 3:
        reward_gap_next = -10000
    else:
        reward_gap_next = 0
    
    return reward_dis + reward_snack + reward_dead + reward_gap_next
    
def action(current_state):
    explore = bernoulli.rvs(eps, size = 1)[0]
    if current_state[6]:
        snake_len = 2
    else :
        snake_len = 1
    if explore :
        return random_move(snake_len, current_state[0])
    else :
        state_counter = state_dict[current_state]
        action_array = np.where(q_mat[state_counter] == max(q_mat[state_counter]))[0]
        action_choice = random.choice(action_array)
        return prev_actions[action_choice]

# # def next_state(current_state, current_action, )
#handling next state through current state functiomn for now
    
# #action will depend on the exploration var and the learned q matrix. In case of 
# # exploration, the action will be selected randomly from the set of possible actions.
    
# # def current_action(eps, q_matrix):
    
# # for starters, the reward can be positive if any of the distance(vertical/horizontal)
# # decreases post the current action and negative if that increases
    
# # def reward_fun():
    
# # def q_learn():
    
    

global scores, steps

scores_l = []
steps_l = []
 
def main():
    global width, rows, s, snack

    win = pygame.display.set_mode((width, width))
    current_move = 1
    last_move = 1
    s = snake((255,0,0), (10,10), width, rows, last_move)
    snack = cube(randomSnack(rows, s), width = width, rows = rows, color=(0,255,0))
    flag = True
    steps = 0
    episodes = 0
    clock = pygame.time.Clock()
    
    while steps <= 10000 and episodes <= 5000:
        pygame.time.delay(10)
        clock.tick(100000)
        act_0, d_x_0, d_y_0, gap_up_0, gap_left_0, gap_right_0, snake_len_f_0 = current_state(last_move, s, snack, rows, 1)
        lst_0 = (act_0, d_x_0, d_y_0, gap_up_0, gap_left_0, gap_right_0, snake_len_f_0)
        key = action(lst_0)
        s.move(key)
        last_move = s.last_move
        act, d_x, d_y, gap_up, gap_left, gap_right, snake_len_f = current_state(s.last_move, s, snack, rows, 1)
        lst = (act, d_x, d_y, gap_up, gap_left, gap_right, snake_len_f)

        print('steps :', steps)
        print('episodes :', episodes)
        steps += 1
        snack_curr = 0
        if s.body[0].pos == snack.pos:
            s.addCube()
            snack_curr = 1
            snack = cube(randomSnack(rows, s), width, rows, color=(0,255,0))
        
        dead = 0
        
        for x in range(len(s.body)):
            if s.body[x].pos in list(map(lambda z:z.pos,s.body[x+1:])):
                dead = 1
                print('Score: ', len(s.body))
                scores_l.append(len(s.body))
                steps_l.append(steps)
                s.reset((10,10))
                episodes += 1
                steps = 0
                break
        
        q_mat = q_matrix_update(lst_0, last_move, lst, snack_curr, dead)
           
        redrawWindow(win)
 
       
    pass
 
 
 
main()

score_df = pd.DataFrame()
score_df['steps'] = steps_l
score_df['score'] = scores_l
score_df['avg_steps_per_point'] = score_df['steps']/score_df['score']
score_df.avg_steps_per_point.plot.line()
score_df.score.plot.line()

def play():
    
    return