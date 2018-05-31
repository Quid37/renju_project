#!/usr/bin/python3.4

import numpy as np 
import time
import tensorflow as tf 
import logging


import sys
import numpy as np


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


MAX_DEPTH_SIMUL = 10

MAX_DEPTH_ROLLOUT = 10

TIME_LIMIT = 15

REWARD_WIN = 1
REWARD_DRAW = 0
REWARD_LOSE = -1

BRD_SZ = 225
TBL_SZ = 675
BRD_SHP = (1, 15, 15)

black_layer = np.ones(BRD_SHP, dtype=np.float32)
white_layer = np.zeros(BRD_SHP, dtype=np.float32)




SL_PATH = './test_model-17.meta'
SL_CHECKPOINT = './'





class State():
    def __init__(self, player):
        self.player = player
        self.actions_array = [None for i in range(225)]
        self.actions_cnt = np.zeros(BRD_SZ, dtype=np.int32)
        self.prior_probs = np.zeros(BRD_SZ, dtype=np.float32)
        self.visits_cnt = 0
        self.q_function = np.zeros(BRD_SZ, dtype=np.float32)



class myMCTS():
    def __init__(self):
        self.hello = "Hello"
        self.table = np.zeros((1, 15, 15, 3))
        self.root = State(0)
        self.start_session()



    def start_session(self):
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(SL_PATH)
        saver.restore(self.sess, tf.train.latest_checkpoint(SL_CHECKPOINT, latest_filename="checkpointUchiha"))
        self.prediction = tf.nn.softmax(tf.get_default_graph().get_tensor_by_name(name="my_prediction:0"))
        self.x_tens = tf.get_default_graph().get_tensor_by_name(name="x:0")


    def give_cur_state(self, step_i, step_j, player_given):
        self.start_session()
        self.player = 1 - player_given

        if step_i == -1:
            self.table[0, :, :, 2] = black_layer
            self.root.player = 0
            self.root.prior_probs = self.make_probabilities(self.table)
        else:
            self.table[0, step_i, step_j, player_given] = 1
            next_action = pos_to_num(step_i, step_j)
            if self.root.actions_cnt[next_action] == 0:
                new_root = State(self.player)
                self.root = new_root
                self.root.prior_probs = self.make_probabilities(self.table)
            else:
                self.root = self.root.actions_array[next_action]
            self.root.player = 1 - player_given

        self.sess.close()


    def make_move(self):
        self.start_session()

        self.traverses_cnt = 0

        self.small_wins_cnt = 0
        self.small_loss_cnt = 0

        not_occup_root = np.ones((15, 15))

        result = 0
        res_i = 0
        res_j = 0
        for i in range(15):
            if result == 1:
                break
            for j in range(15):
                if self.table[0,i,j,0] == 1 or self.table[0,i,j,1] == 1:
                    not_occup_root[i, j] = 0
                    continue

                self.table[0, i, j, self.player] = 1
                result = check_win_particular(self.table, i, j, self.player)
                self.table[0, i, j, self.player] = 0

                if result == 1:
                    res_i = i
                    res_j = j
                    break

        if result == 1:
            self.table[0, res_i, res_j, self.root.player] = 1

            self.root.player = 1 - self.root.player
            next_action = pos_to_num(res_i, res_j)
            if self.root.actions_cnt[next_action] == 0:
                new_root = State(self.player)
                self.root = new_root
                self.root.prior_probs = self.make_probabilities(self.table)
            else:
                self.root = self.root.actions_array[next_action]

            return res_i, res_j

        not_occup_root = np.reshape(not_occup_root, (225,))

        result = 0
        res_i = 0
        res_j = 0
        for i in range(15):
            if result == 1:
                break
            for j in range(15):
                if self.table[0,i,j,0] == 1 or self.table[0,i,j,1] == 1:
                    continue

                self.table[0, i, j, 1 - self.player] = 1
                result = check_win_particular(self.table, i, j, 1 - self.player)
                self.table[0, i, j, 1 - self.player] = 0

                if result == 1:
                    res_i = i
                    res_j = j
                    break

        if result == 1:
            self.table[0, res_i, res_j, self.root.player] = 1

            self.root.player = 1 - self.root.player
            next_action = pos_to_num(res_i, res_j)
            if self.root.actions_cnt[next_action] == 0:
                new_root = State(self.player)
                self.root = new_root
                self.root.prior_probs = self.make_probabilities(self.table)
            else:
                self.root = self.root.actions_array[next_action]

            return res_i, res_j


        timer = time.time()
        while (time.time() - timer) < (TIME_LIMIT - 1):
            self.run_MCTS(self.table)
            self.traverses_cnt += 1
            self.player = self.root.player


        logging.debug("traverses counter:", self.traverses_cnt)
        logging.debug(self.root.q_function)

        self.sess.close()
        logging.debug(self.root.actions_cnt)
        logging.debug(self.small_wins_cnt, " - ", self.small_loss_cnt)

        move_i = 0
        move_j = 0

        important = self.root.actions_cnt > 120
        if np.max(important * self.root.q_function / (1 + self.root.actions_cnt)) > 0.5:
            move_i, move_j = num_to_pos(np.argmax(not_occup_root + important / (1 + self.root.actions_cnt)))
        elif np.max(self.root.q_function) <= 0:
            move_i, move_j = num_to_pos(np.argmax(self.root.prior_probs))
        else:
            move_i, move_j = num_to_pos(np.argmax(not_occup_root + \
                                         (self.root.actions_cnt / (1 + self.root.visits_cnt) + \
                                          self.root.q_function / (1 + self.root.actions_cnt))))

        self.table[0, move_i, move_j, self.root.player] = 1

        self.root.player = 1 - self.root.player
        next_action = pos_to_num(move_i, move_j)
        if self.root.actions_cnt[next_action] == 0:
            new_root = State(self.player)
            self.root = new_root
            self.root.prior_probs = self.make_probabilities(self.table)
        else:
            self.root = self.root.actions_array[next_action]

        return move_i, move_j




    def run_MCTS(self, table_given):
        table = np.copy(table_given)

        cur_vertex = self.root
        self.depth = 0

        trav_path = []
        leaf_found = 0

        not_occupied = np.reshape(1 - np.add(table[0,:,:,0], table[0,:,:,1]), (225))

        while leaf_found == 0 and self.depth < MAX_DEPTH_SIMUL:
            if self.player == 0:
                table[:,:,:,2] = black_layer
            else:
                table[:,:,:,2] = white_layer

            
            result = 0
            res_i = 0
            res_j = 0

            next_action = None

            if result == 0:

                updated_probs =  np.argmax(not_occupied + (cur_vertex.actions_cnt > 60) * (cur_vertex.q_function > (cur_vertex.actions_cnt * 0.55)) + \
                        ((7 * cur_vertex.prior_probs + cur_vertex.q_function) * not_occupied) / (1 + 2 * cur_vertex.actions_cnt))

                next_action = updated_probs

            else:
                next_action = pos_to_num(res_i, res_j)


            next_i, next_j = num_to_pos(next_action)

            not_occupied[next_action] = 0
            table[0, next_i, next_j, self.player] = 1

            if cur_vertex.actions_cnt[next_action] == 0:
                next_vertex = State(1 - self.player)
                next_vertex.prior_probs = self.make_probabilities(table)
                cur_vertex.actions_array[next_action] = next_vertex
                leaf_found = 1

            trav_path.append(next_action)

            cur_vertex = cur_vertex.actions_array[next_action]

            
            self.player = 1 - self.player
            self.depth += 1


        winner = self.run_rollout(table)

        self.back_propagation(winner, trav_path)




    def run_rollout(self, table_given):
        not_occupied = np.reshape(np.add(black_layer, -np.add(table_given[0,:,:,0], table_given[0,:,:,1])), (225))

        finished = 0


        while finished == 0 and self.depth < MAX_DEPTH_ROLLOUT:
            if self.player == 0:
                table_given[:,:,:,2] = black_layer
            else:
                table_given[:,:,:,2] = white_layer

            step_prediction = self.make_probabilities(table_given)[0]
            
            step_prediction *= not_occupied
            step_prediction /= np.sum(step_prediction)
            step_prediction = softmax(step_prediction)
            

            next_step_i = 0
            next_step_j = 0
            if np.max(step_prediction) < 0.15:
                next_step_i, next_step_j = num_to_pos(np.argmax(step_prediction))
            else:
                next_step_i, next_step_j = num_to_pos(np.random.choice(BRD_SZ, p=step_prediction))

            table_given[0, next_step_i, next_step_j, self.player] = 1
            not_occupied[pos_to_num(next_step_i, next_step_j)] = 0

            if check_win_particular(table_given, next_step_i, next_step_j, self.player) == 1:
                finished = 1
                break


            self.player = 1 - self.player
            self.depth += 1

        if finished == 1:
            return self.player
        else:
            return -1


    def back_propagation(self, winner, trav_path):
        state = self.root
        if self.root.player == winner:
            self.small_wins_cnt += 1
        elif winner != -1:
            self.small_loss_cnt += 1
        for cur in range(0, len(trav_path)):
            action = trav_path[cur]

            if winner != -1:
                if state.player == winner:
                    state.q_function[action] += REWARD_WIN
                else:
                    state.q_function[action] += REWARD_LOSE
            else:
                state.q_function[action] += REWARD_DRAW

            state.visits_cnt += 1
            state.actions_cnt[action] += 1
            state = state.actions_array[action]

        return
            

    def make_probabilities(self, table_given):
        return self.sess.run(self.prediction, feed_dict={self.x_tens : table_given})

            
            


def softmax(mtrx):
    e_x = np.exp(mtrx - np.max(mtrx))
    return e_x / e_x.sum()


def pos_to_num(i, j):
    return i * 15 + j

def num_to_pos(i):
    return i // 15, i % 15
    
def check_win_particular(table, st_i, st_j, player):
    for i in range(max(0, st_i - 4), min(11, st_i + 1)):
        if np.sum(table[0, i:i+5, st_j, player]) == 5:
            return 1

    for j in range(max(0, st_j - 4), min(11, st_j + 1)):
        if np.sum(table[0, st_i, j:j+5, player]) == 5:
            return 1

    summ = 0

    for i in range(-4, 1):
        if (st_i + i) >= 0 and (st_j + i) >= 0 and (st_i + i) < 15 and (st_j + i) < 15:
            for j in range(5):
                if st_i + i + j < 15 and st_j + i + j < 15:
                    summ += table[0, st_i + i + j, st_j + i + j, player]
                else:
                    summ = 0
                    break
            if summ == 5:
                return 1
            summ = 0

    summ = 0

    for i in range(0, 5):
        if (st_i + i) >= 0 and (st_j - i) >= 0 and (st_i + i) < 15 and (st_j - i) < 15:
            for j in range(5):
                if st_i + i - j >= 0 and st_j - i + j < 15:
                    summ += table[0, st_i + i - j, st_j - i + j, player]
                else:
                    summ = 0
                    break
            if summ == 5:
                return 1
            summ = 0

    return 0



def wait_for_game_update():
    if not sys.stdin.closed:
        game_dumps = sys.stdin.readline()
        if game_dumps:
            return game_dumps
    return None

def set_move(move):
    if sys.stdout.closed:
        return False
    sys.stdout.write(move + '\n')
    sys.stdout.flush()
    return True

def interp_step(step):
    ns = [0, 0]
    ns[0] = ord(step[0]) - ord('a')
    if (ord(step[0]) > ord('i')):
        ns[0] -= 1
    ns[1] = int(step[1:]) - 1
    return ns


letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p']



def main():
    UchihaPower = myMCTS()

    the_board = np.zeros((15, 15, 3))
    black_board = np.ones((15, 15))
    white_board = np.zeros((15, 15))

    pid = os.getpid()

    color = 1

    try:
        while True:
            print_board(the_board)

            logging.debug("Wait for game update...")
            game = wait_for_game_update()

            if not game:
                logging.debug("Game is over!")
                return

            if game == "\n":
                color = 0
                UchihaPower.give_cur_state(-1, -1, 1 - color)
                the_board[:,:,2] = black_board

            next_move = game.split()


            if game != "\n":
                next_step = interp_step(next_move[-1])
                the_board[next_step[0], next_step[1], 1 - color] = 1
                UchihaPower.give_cur_state(next_step[0], next_step[1], 1 - color)

            move_i, move_j = UchihaPower.make_move()

            tf.reset_default_graph()

            the_board[move_i, move_j, color] = 1

            go_j = move_j + 1
            go_i = letters[move_i]

            made_move = game[:-1]
            if game != "\n":
                made_move += " "
            made_move += (str(go_i) + str(go_j))

            if not set_move(made_move):
                logging.error("Impossible set move!")
                return

            logging.debug('my move: %s', made_move)
    except:
        logging.error('Error!', exc_info=True, stack_info=True)



if __name__ == "__main__":
    main()
