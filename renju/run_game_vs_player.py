import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
from numpy import unravel_index
import time


table = np.zeros([15, 15, 3], dtype=np.float32)

W_conv = tf.get_variable("W_conv", shape=[5, 5, 1, 20])

sess = tf.Session()
saver = tf.train.import_meta_graph('./test_model-17.meta')
saver.restore(sess, tf.train.latest_checkpoint('./', latest_filename="checkpointUchiha"))




def sorted_idx_possible(preds, cur_WIDTH, table):
	predstr = np.copy(np.reshape(preds, (225)))
	idx = np.argsort(-predstr)
	maxidx = np.empty([cur_WIDTH, 2], dtype = np.int32)
	cnt = 0
	pos = 0
	probabilities = np.empty((cur_WIDTH), dtype=np.float32)

	while cnt < cur_WIDTH:
		while pos < 225 and not (table[idx[pos] // 15, idx[pos] % 15, 0] == 0 and table[idx[pos] // 15, idx[pos] % 15, 1] == 0):
			pos += 1

		if pos == 225:
			print(table[:,:,0])
			print(table[:,:,1])

		maxidx[cnt, 0] = idx[pos] // 15
		maxidx[cnt, 1] = idx[pos] % 15
		probabilities[cnt] = predstr[idx[pos]]
		cnt += 1
		pos += 1

	return maxidx, probabilities




def check_win():
	for i in range(11):
		for j in range(15):
			sum = np.sum(table[i:i+5, j, 0])
			if sum == 5:
				return 1
			sum = np.sum(table[i:i+5, j, 1])
			if sum == 5:
				return -1

	for i in range(15):
		for j in range(11):
			sum = np.sum(table[i, j:j+5, 0])
			if sum == 5:
				return 1
			sum = np.sum(table[i, j:j+5, 1])
			if sum == 5:
				return -1

	for i in range(11):
		for j in range(11):
			sum = 0
			for k in range(5):
				sum += table[i + k, j + k, 0]
			if sum == 5:
				return 1
			sum = 0
			for k in range(5):
				sum += table[i + k, j + k, 1]
			if sum == 5:
				return -1

	for i in range(4, 15):
		for j in range(11):
			sum = 0
			for k in range(5):
				sum += table[i - k, j + k, 0]
			if sum == 5:
				return 1
			sum = 0
			for k in range(5):
				sum += table[i - k, j + k, 1]
			if sum == 5:
				return -1

	return 0

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']

def print_board():
	print()
	print("  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15")
	for i in range(15):
		print(letters[i], end=" ")
		for j in range(15):
			if table[i, j, 0] == 0 and table[i, j, 1] == 0:
				print('. ', end=" ")
			elif table[i, j, 0] == 1:
				print('X ', end=" ")
			else:
				print('O ', end=" ")
		print()
	print()


prediction_raw = tf.get_default_graph().get_tensor_by_name(name="my_prediction:0")
prediction = tf.reshape(tf.nn.softmax(prediction_raw), [-1, 15 * 15])


print("choose your side")
side = input()

if (side == "black" or side == "b" or side == "B" or side == "1" or side == "b_term"):

	step = 1
	x_tens = tf.get_default_graph().get_tensor_by_name(name="x:0")

	if (side != "b_term"):
		print("you play for black")

	while check_win() == 0:

		if (side != "b_term"):
			print_board()

		if (step % 2 == 1):
			if (side != "b_term"):
				print("your turn")

			x_char = input()
			x = ord(x_char[0]) - ord('a')
			y = int(x_char[1:])
			y -= 1
			if x < 0 or x >= 15 or y < 0 or y >= 15 or table[x, y, 0] != 0:
				print("invalid step\n")
				continue
			table[x, y, 0] = 1
			
		else:
			pred_np = np.empty((15, 15))
			time00 = time.time()
			pred_np = sess.run(prediction, feed_dict={x_tens: np.reshape(table, (1, 15, 15, 3))})
			pred_np = np.reshape(pred_np, [15, 15])
			print(pred_np)
			max_i = -1
			max_j = -1
			for i in range(15):
				for j in range(15):
					if (max_i == -1 and table[i, j, 1] == 0 and table[i, j, 0] == 0) or \
								(max_i != -1 and table[i, j, 1] == 0 and table[i, j, 0] == 0 and pred_np[i, j] > pred_np[max_i, max_j]):
								max_i = i
								max_j = j

			if (side != "b_term"):
				print("enemy: ", end="")
			print(letters[max_i], max_j + 1)
			
			table[max_i, max_j, 1] = 1

		step += 1

	print_board()
	if check_win() == 1:
		print("YOU WIN")
		exit(1)
	else:
		print("YOU LOSE")
		exit(2)
else:
	step = 0
	x_tens = tf.get_default_graph().get_tensor_by_name(name="x:0")
	table[:,:,2] = np.ones((15, 15))

	if (side != "w_term"):
		print("you play for white")

	while check_win() == 0:

		print_board()
		if (step % 2 == 1):
			if (side != "w_term"):
				print("your turn")

			x_char = input()
			x = ord(x_char[0]) - ord('a')
			y = int(x_char[1:])
			y -= 1
			if x < 0 or x >= 15 or y < 0 or y >= 15 or table[x, y, 0] != 0:
				print("invalid step\n")
				continue
			table[x, y, 1] = 1
			
		else:
			pred_np = np.empty((15, 15))
			pred_np = np.reshape(sess.run(prediction, feed_dict={x_tens: np.reshape(table, (1, 15, 15, 3))}), (15, 15))
			print(pred_np)

			possible_turn, probabilities = sorted_idx_possible(pred_np, 4, table)
			pos = np.random.choice(4, p=(probabilities / np.sum(probabilities)))
			step_i = possible_turn[pos, 0]
			step_j = possible_turn[pos, 1]
			print(possible_turn)
			print(probabilities)
			print("i would chose", step_i, step_j)
			print()


			max_i = -1
			max_j = -1
			for i in range(15):
				for j in range(15):
					if (max_i == -1 and table[i, j, 1] == 0 and table[i, j, 0] == 0) or \
								(max_i != -1 and table[i, j, 1] == 0 and table[i, j, 0] == 0 and pred_np[i, j] > pred_np[max_i, max_j]):
								max_i = i
								max_j = j
			
			if (side != "w_term"):
				print("enemy: ", end="")
			print(letters[max_i], max_j + 1)
			
			table[max_i, max_j, 0] = 1

		step += 1

	print_board()
	if check_win() == -1:
		print("YOU WIN")
		exit(1)
	else:
		print("YOU LOSE")
		exit(2)

