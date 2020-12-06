
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

def random_move(snake_len, last_move):
    actions = [1, -1, 10, -10]
    if snake_len < 2:
        move = random.choice(actions)
    else:
        actions2 = [x for x in actions if x != -1 * last_move]
        move = random.choice(actions2)
    return move

def snake_mat(rows,snake):
    mat = np.zeros((rows, rows))
    for cube in snake.body:
        mat[cube.pos[0]][cube.pos[1]] = 1
    return mat
  
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
    
    dis_x_a = np.abs(snake_head_pos[0] - snack_pos[0])
    dis_y_a = np.abs(snake_head_pos[1] - snack_pos[1])
    
    dis_x_b = np.abs(snake_head_pos[0] + rows - snack_pos[0])
    dis_y_b = np.abs(snake_head_pos[1] + rows - snack_pos[1])
    
    dis_x = np.min(dis_x_a, dis_x_b)
    dis_y = np.min(dis_y_a, dis_y_b)
    
    dis_x = snake_dis_bucket(dis_x, dis_xs)
    dis_y = snake_dis_bucket(dis_y, dis_ys)
    
    snake_len = len(snake.body)
    snake_len_f = snake_len_bucket(snake_len)
    gap_up = 0
    gap_left = 0
    gap_right = 0
    
    horz_gap = horizontal_gap_snake_snack(snake, snack_pos)
    ver_gap = vertical_gap_snake_snack(snake, snack_pos)
    
    if len(snake.body) > 2:
        dir_x = x_pos - snake.body[1].pos[0]
        dir_y = y_pos - snake.body[1].pos[1]
        
        snake_pos_mat = snake_mat(rows, snake)
        
        gap_up = snake_gap_up(x_pos, y_pos, dir_x, dir_y, k, snake_pos_mat)
        gap_left = snake_gap_left(x_pos, y_pos, dir_x, dir_y, k, snake_pos_mat)
        gap_right = snake_gap_right(x_pos, y_pos, dir_x, dir_y, k, snake_pos_mat)
        
    return dis_x, dis_y, gap_up, gap_left, gap_right, snake_len_f, horz_gap, ver_gap


def snake_len_bucket(snake_len) :
    array = np.asarray(snake_len_v)
    index = (np.abs(array - snake_len + 1)).argmin()
    return array[index]

def horizontal_gap_snake_snack(snake, snack_pos) :
    snake_head_pos = snake.body[0].pos
    snake_head_x = snake_head_pos[0]
    snack_head_x = snack_pos[0]
    
    if len(snake.body) < 2 :
        return 0
    
    else :
        index_list = [x.pos[0] for x in snake.body]
        index_list.pop(0)
        
        sum1 = sum([(x > snack_head_x) and (x < snake_head_x) for x in index_list])
        sum2 = sum([(x < snack_head_x) and (x > snake_head_x) for x in index_list])
        
        return (sum1 + sum2) > 0
    
def vertical_gap_snake_snack(snake, snack_pos) :
    snake_head_pos = snake.body[0].pos
    snake_head_y = snake_head_pos[1]
    snack_head_y = snack_pos[1]
    
    if len(snake.body) < 2 :
        return 0
    
    else :
        index_list = [y.pos[1] for y in snake.body]
        index_list.pop(0)
        
        sum1 = sum([(y > snack_head_y) and (y < snake_head_y) for y in index_list])
        sum2 = sum([(y < snack_head_y) and (y > snake_head_y) for y in index_list])
        
        return (sum1 + sum2) > 0
    
def snake_dis_bucket(dis, array):
    index = (np.abs(array - dis)).argmin()
    return array[index]
    
eps = 0.5
alpha = 0.7
gamma = 0.9
lambda_z = 0.8

state_counter = 0
state_dict = {}
rows = 10
width = 200

prev_actions = [1, -1, 10, -10]
dis_xs = np.arange(0, rows, 2)
dis_ys = np.arange(0, rows, 2)
gaps_up = [0, 1]
gaps_left = [0, 1]
gaps_right = [0, 1]
snake_len_v = [0, 10, 20]
horz_gap = [0, 1]
vert_gap = [0, 1]

for b in dis_xs:
    for c in dis_ys:
        for d in gaps_up:
            for e in gaps_left:
                for f in gaps_right:
                    for g in snake_len_v:
                        for h in horz_gap:
                            for i in vert_gap: 
                                state_dict[b,c,d,e,f,g,h,i] = state_counter
                                state_counter += 1
        
global q_mat, z_trace
q_mat = np.zeros((len(state_dict), len(prev_actions)))
z_trace = np.zeros((len(state_dict), len(prev_actions)))
                        
def q_matrix_update(current_state, current_action, next_state, snack, dead, z_trace):
    state_counter = state_dict[current_state]
    state_counter_next = state_dict[next_state]
    current_action_index = prev_actions.index(current_action)
    reward = reward_function(current_state, next_state, snack, dead)
    max_reward_next_state = np.max(q_mat[state_counter_next])
    z_trace[state_counter, current_action_index] = 1
    delta = (reward + gamma * max_reward_next_state - q_mat[state_counter, current_action_index])
    for i in range(0, q_mat.shape[0]):
        for j in range(0, q_mat.shape[1]):
           q_mat[i, j] += delta * alpha * z_trace[i, j]
           z_trace[i, j] = gamma * lambda_z * z_trace[i, j]
    
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
        reward_dis = -1000 
    else :
        reward_dis = 100 
    
    if snack == 1:
        reward_snack = 1000
    else :
        reward_snack = 0
        
    if dead == 1 :
        reward_dead = -10000
    else :
        reward_dead = 0
        
    # if gap_up_next + gap_left_next + gap_right_next == 2:
    #     reward_gap_next = -100
    # elif gap_up_next + gap_left_next + gap_right_next == 3:
    #     reward_gap_next = -10000
    # else:
    #     reward_gap_next = 0
    
    return reward_dis + reward_snack + reward_dead #+ reward_gap_next
    
def action(current_state, last_move, snake_len_actual):
    explore = bernoulli.rvs(eps, size = 1)[0]
    index_neg_last_action = prev_actions.index(-1 * last_move)
    if current_state[6]:
        snake_len = 2
    else :
        snake_len = 1
    if explore :
        # print('action1')
        return random_move(snake_len_actual, last_move)
        
    else :
        state_counter = state_dict[current_state]
        action_array = np.where(q_mat[state_counter] == max(q_mat[state_counter]))[0]
        
        if snake_len_actual < 2 :
            if len(action_array) > 1 :
                action_choice = random.choice(action_array)
                # print('action2')
                return prev_actions[action_choice]
            else:
                # print('action3')
                return prev_actions[action_array[0]]
        else:
            if index_neg_last_action in action_array :
                action_list = list(action_array)
                action_list.remove(index_neg_last_action)
                if len(action_list) == 0 :
                    lst = list(q_mat[state_counter])
                    lst_cpy = set(lst)
                    lst_cpy.remove(max(lst_cpy))
                    new_max = max(lst_cpy)
                    new_action_array = np.where(q_mat[state_counter] == new_max)[0]
                    action_choice = random.choice(new_action_array)
                    return prev_actions[action_choice]
                else:
                    # print('action4')
                    action_choice = random.choice(action_list)
                    return prev_actions[action_choice]
            else:
                # print('action5')
                action_choice = random.choice(action_array)
                return prev_actions[action_choice]
 

global scores, steps

scores_l = []
steps_l = []
 
def main():
    global width, rows, s, snack, eps, q_mat, scores_l, steps_l, z_trace, quit_window
    win = pygame.display.set_mode((width, width))
    current_move = 1
    last_move = 1
    s = snake((255,0,0), (10,10), width, rows, last_move)
    snack = cube(randomSnack(rows, s), width = width, rows = rows, color=(0,255,0))
    steps = 0
    episodes = 0
    clock = pygame.time.Clock()
    quit_window = 0
    
    while episodes <= 1500:
        pygame.time.delay(10)
        clock.tick(500000)
        
        d_x_0, d_y_0, gap_up_0, gap_left_0, gap_right_0, snake_len_f_0, horz_gap_0, ver_gap_0 = current_state(last_move, s, snack, rows, 1)
        lst_0 = (d_x_0, d_y_0, gap_up_0, gap_left_0, gap_right_0, snake_len_f_0, horz_gap_0, ver_gap_0)
        key = action(lst_0, last_move, len(s.body))
        s.move(key)
        if last_move == -1* s.last_move and len(s.body) > 1:
            print('Error')
            print('actual snake length :', len(s.body))
        last_move = s.last_move
        d_x, d_y, gap_up, gap_left, gap_right, snake_len_f, horz_gap, ver_gap = current_state(s.last_move, s, snack, rows, 1)
        lst = (d_x, d_y, gap_up, gap_left, gap_right, snake_len_f, horz_gap, ver_gap)

        print('steps :', steps)
        print('episodes :', episodes)
        print('epsilon :',eps)
        steps += 1
        snack_curr = 0
        if s.body[0].pos == snack.pos:
            s.addCube()
            snack_curr = 1
            snack = cube(randomSnack(rows, s), width, rows, color=(0,255,0))
        
        dead = 0
        
        for x in range(len(s.body)):
            if s.body[x].pos in list(map(lambda z:z.pos,s.body[x+1:])) or steps > 10000 :
                dead = 1
                print('Score: ', len(s.body))
                scores_l.append(len(s.body))
                steps_l.append(steps)
                s.reset((10,10))
                episodes += 1
                if episodes > 0 and episodes % 100 == 0 and eps > 0.1 :
                    eps = eps - 0.04
                steps = 0
                break
        
        q_mat = q_matrix_update(lst_0, last_move, lst, snack_curr, dead, z_trace)
        if dead == 1 :
            z_trace = np.zeros((len(state_dict), len(prev_actions)))
           
        redrawWindow(win)
    quit_window = 1
    pass
 
main()

score_df = pd.DataFrame()
score_df['steps'] = steps_l
score_df['score'] = scores_l
score_df['avg_steps_per_point'] = score_df['steps']/score_df['score']
score_df.score.plot.line()

scores_f = []
steps_f = []
score_df_play = pd.DataFrame()

def play(n_episodes, speed, delay, max_steps):
    global width, rows, s, snack, q_mat, scores_f, steps_f, score_df_play
    win = pygame.display.set_mode((width, width))
    current_move = 1
    last_move = 1
    s = snake((255,0,0), (10,10), width, rows, last_move)
    snack = cube(randomSnack(rows, s), width = width, rows = rows, color=(0,255,0))
    steps = 0
    episodes = 0
    clock = pygame.time.Clock()
    
    while episodes <= n_episodes:
        pygame.time.delay(delay)
        clock.tick(speed)
        
        d_x_0, d_y_0, gap_up_0, gap_left_0, gap_right_0, snake_len_f_0, horz_gap_0, ver_gap_0 = current_state(last_move, s, snack, rows, 1)
        lst_0 = (d_x_0, d_y_0, gap_up_0, gap_left_0, gap_right_0, snake_len_f_0, horz_gap_0, ver_gap_0)
        key = action(lst_0, last_move, len(s.body))
        s.move(key)
        if last_move == -1 * s.last_move and len(s.body) > 1:
            print('Error')
            print('actual snake length :', len(s.body))
        last_move = s.last_move
        print('steps :', steps)
        print('episodes :', episodes)
        print('epsilon :',eps)
        
        steps += 1
        snack_curr = 0
        if s.body[0].pos == snack.pos:
            s.addCube()
            snack_curr = 1
            snack = cube(randomSnack(rows, s), width, rows, color=(0,255,0))
        
        for x in range(len(s.body)):
            if s.body[x].pos in list(map(lambda z:z.pos,s.body[x+1:])) or steps > max_steps:
                print('Score: ', len(s.body))
                scores_f.append(len(s.body))
                steps_f.append(steps)
                s.reset((10,10))
                episodes += 1
                steps = 0
                break
        
        redrawWindow(win)
    
    score_df_play['steps'] = steps_f
    score_df_play['score'] = scores_f
    score_df_play.score.plot.line()
    
    pass
