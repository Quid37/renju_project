import numpy as np
import time
import tensorflow as tf
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = ''


fl = open("train-white.renju", "r")

all_white_lines = fl.readlines()

lines_wh = np.array(all_white_lines[900000:])
whsz = lines_wh.shape[0]

fl.close()
fl = open("train-black.renju", "r")

all_black_lines = fl.readlines()

lines_bl = np.array(all_black_lines[1000000:])
fl.close()
blsz = lines_bl.shape[0]


test_features = np.empty((whsz + blsz, 15, 15, 3), dtype = np.float32)
test_labels = np.zeros((whsz + blsz, 15, 15), dtype = np.float32)
applied = 0



def shift_reverse(feat, lbl):
	need_change = np.random.uniform()
	if need_change < 0.1:
		return feat, lbl


	new_feat = np.copy(feat)
	new_lbl = np.copy(lbl)

	rot_num = np.random.randint(0, high=4)
	new_feat = np.rot90(new_feat, rot_num)
	new_lbl = np.rot90(new_lbl, rot_num)


	min_i = 15
	max_i = -1
	min_j = 15
	max_j = -1
	for i in range(15):
		for j in range(15):
			if (new_feat[i,j,0] == 1 or new_feat[i,j,1] == 1 or new_lbl[i,j] == 1):
				if i < min_i:
					min_i = i
				if i > max_i:
					max_i = i
				if j < min_j:
					min_j = j
				if j > max_j:
					max_j = j

	shift_gap_i = min_i + (15 - max_i)
	shift_gap_j = min_j + (15 - max_j)

	shift_i = np.random.randint(0, high=shift_gap_i)
	shift_j = np.random.randint(0, high=shift_gap_j)

	if shift_i > min_i:
		shift_i -= shift_gap_i
		shift_i -= 1
	
	if shift_j > min_j:
		shift_j -= shift_gap_j
		shift_j -= 1


	new_feat = np.roll(new_feat, 15-shift_j, axis=1)
	new_lbl = np.roll(new_lbl, 15-shift_j, axis=1)

	new_feat = np.roll(new_feat, 15-shift_i, axis=0)
	new_lbl = np.roll(new_lbl, 15-shift_i, axis=0)

	return new_feat, new_lbl



def interp_step(step):
	ns = [0, 0]
	ns[0] = ord(step[0]) - ord('a')
	if (ord(step[0]) > ord('i')):
		ns[0] -= 1
	ns[1] = int(step[1:]) - 1
	return ns


for i in range(whsz):
	board = np.zeros((15, 15, 3))
	game = np.array(lines_wh[i].split()[1:])
	turn_to_stop = np.random.randint(game.shape[0])
	if (game.shape[0] % 2 == 1) and (turn_to_stop == (game.shape[0] - 1)):
		turn_to_stop -= 1
	turn_to_stop = turn_to_stop // 2
	turn = 0
	while True:
		step_b = interp_step(game[turn * 2])
		board[step_b[0], step_b[1], 0] = 1

		step_w = interp_step(game[turn * 2 + 1])
		if turn == turn_to_stop:
			newlabel = np.zeros((15, 15))
			newlabel[step_w[0], step_w[1]] = 1
			test_features[applied] = board
			test_labels[applied] =  newlabel
			applied += 1
			break
		else:
			board[step_w[0], step_w[1], 1] = 1

		turn += 1

for i in range(blsz):
	board = np.zeros((15, 15, 3))
	board[:,:,2] = np.ones((15, 15)) 
	game = np.array(lines_bl[i].split()[1:])
	turn_to_stop = np.random.randint(game.shape[0])
	turn_to_stop = turn_to_stop // 2
	turn = 0
	while True:
		step_b = interp_step(game[turn * 2])
		if turn == turn_to_stop:
			newlabel = np.zeros((15, 15))
			newlabel[step_b[0], step_b[1]] = 1
			test_features[applied] = board
			test_labels[applied] = newlabel
			applied += 1
			break
		else:
			board[step_b[0], step_b[1], 0] = 1

		step_w = interp_step(game[turn * 2 + 1])
		board[step_w[0], step_w[1], 1] = 1

		turn += 1

'''
ftrs_file = open("test_features", "w")
lbls_file = open("test_labels", "w")

for i in range(whsz + blsz):
	ftrs_file.write(test_features[i])
'''

GEN_BL = 200
GEN_WH = 180
TRAV_LEN = 5000

generate_step = 0
CUR_WH = 0
CUR_BL = 0

train_black_games = all_black_lines[:1000000]
train_white_games = all_white_lines[:900000]

def feed_generator(multip):
	global generate_step
	global CUR_WH
	global CUR_BL


	ret_features = np.empty((((GEN_BL + GEN_WH) * multip), 15, 15, 3), dtype=np.float32)
	ret_labels = np.empty((((GEN_BL + GEN_WH) * multip), 15, 15), dtype=np.float32)
	return_cnt = 0


	for i in range(CUR_WH, CUR_WH + GEN_WH * multip):
		board = np.zeros((15, 15, 3))
		game = np.array(train_white_games[i].split()[1:])
		if game.shape[0] == 1:
			continue
		turn_to_stop = np.random.randint(game.shape[0])
		if (turn_to_stop == (game.shape[0] - 1)) and ((game.shape[0] % 2) == 1):
			turn_to_stop -= 1
		turn_to_stop = turn_to_stop // 2
		turn = 0
		while True:
			step_b = interp_step(game[turn * 2])
			board[step_b[0], step_b[1], 0] = 1

			step_w = interp_step(game[turn * 2 + 1])
			if turn == turn_to_stop:
				newlabel = np.zeros((15, 15))
				newlabel[step_w[0], step_w[1]] = 1

				board_aug, newlabel_aug = shift_reverse(board, newlabel)

				ret_features[return_cnt] = board_aug
				ret_labels[return_cnt] = newlabel_aug
				return_cnt += 1
				break
			else:
				board[step_w[0], step_w[1], 1] = 1

			turn += 1


	for i in range(CUR_BL, CUR_BL + GEN_BL * multip):
		board = np.zeros((15, 15, 3))
		board[:,:,2] = np.ones((15, 15)) 
		game = np.array(train_black_games[i].split()[1:])
		#if game.shape[0] == 0:
		#	continue
		turn_to_stop = np.random.randint(game.shape[0])
		turn_to_stop = turn_to_stop // 2
		turn = 0
		while True:
			step_b = interp_step(game[turn * 2])
			if turn == turn_to_stop:
				newlabel = np.zeros((15, 15))
				newlabel[step_b[0], step_b[1]] = 1

				board_aug, newlabel_aug = shift_reverse(board, newlabel)

				ret_features[return_cnt] = board_aug
				ret_labels[return_cnt] = newlabel_aug
				return_cnt += 1
				break
			else:
				board[step_b[0], step_b[1], 0] = 1

			step_w = interp_step(game[turn * 2 + 1])
			board[step_w[0], step_w[1], 1] = 1

			turn += 1

	
	CUR_BL += GEN_BL * multip
	CUR_WH += GEN_WH * multip

	return [ret_features, ret_labels]


EPOCH_LEN = 50000
EPOCH_NUM = 100

TEST_SIZE = 500


def weight_var(shape):
	init = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init)

def bias_var(shape):
	#init = tf.constant(0.1, shape=shape)
	init = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 3, 3, 1],
                             strides= [1, 2, 2, 1], padding='SAME')

def max_pool_2x2_2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                             strides= [1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 15, 15, 3], name="x")
y_ = tf.placeholder(tf.float32, [None, 225])


W_conv1 = weight_var([5, 5, 3, 64])
b_conv1 = bias_var([64])
conv_relu1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)


W_conv2 = weight_var([3, 3, 64, 64])
b_conv2 = bias_var([64])
conv_relu2 = tf.nn.relu(conv2d(conv_relu1, W_conv2) + b_conv2)


W_conv3 = weight_var([3, 3, 64, 128])
b_conv3 = bias_var([128])
conv_relu3 = tf.nn.relu(conv2d(conv_relu2, W_conv3) + b_conv3)


W_conv4 = weight_var([3, 3, 128, 128])
b_conv4 = bias_var([128])
conv_relu4 = tf.nn.relu(conv2d(conv_relu3, W_conv4) + b_conv4)


W_conv5 = weight_var([3, 3, 128, 196])
b_conv5 = bias_var([196])
conv_relu5 = tf.nn.relu(conv2d(conv_relu4, W_conv5) + b_conv5)

W_conv6 = weight_var([2, 2, 196, 196])
b_conv6 = bias_var([196])
conv_relu6 = tf.nn.relu(conv2d(conv_relu5, W_conv6) + b_conv6)


W_conv7 = weight_var([2, 2, 196, 256])
b_conv7 = bias_var([256])
conv_relu7 = tf.nn.relu(conv2d(conv_relu6, W_conv7) + b_conv7)


W_conv8 = weight_var([1, 1, 256, 1])
predict_ = conv2d(conv_relu7, W_conv8)



#NUMBER OF PARAMETERS = 840K
'''
total_param = 0
for variable in tf.trainable_variables():
	shape = variable.get_shape()
	print(shape)
	print(len(shape))
	variable_parameters = 1
	for dim in shape:
		print(dim)
		variable_parameters *= dim.value
	print(variable_parameters)
	total_param += variable_parameters

print("total parameters:", total_param)
exit()
'''

prediction = tf.reshape(tf.reshape(predict_, [-1, 15, 15]), [-1, 225], name="my_prediction")
cross_entropy = tf.reduce_mean(
					tf.nn.softmax_cross_entropy_with_logits(labels = y_,
									logits = prediction))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()

time0 = time.time()

correct_prediction = tf.equal(tf.argmax(prediction, 1),
								tf.argmax(y_, -1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


for epoch in range(EPOCH_NUM):
	cur_multiplier = 1
	if epoch > 25:
		cur_multiplier = 2
	if epoch > 55:
		cur_multiplier = 3

	print("epoch num:", epoch)

	while True:
		if generate_step + cur_multiplier >= TRAV_LEN:
			cur_multipplier = 1

		batch = feed_generator(cur_multiplier)

		batch[1] = np.reshape(batch[1], (batch[1].shape[0], 225))
		sess.run(train_step, feed_dict={x: batch[0], y_:batch[1]})
		if (generate_step % 2400 == 0):
			print("step num:", generate_step + 1)
			print("time:", time.time() - time0)
			#print("cross_entropy:", sess.run(cross_entropy, feed_dict={x: batch[0], y_: batch[1]}))
			#print()
			random_subarr = np.random.randint(whsz + blsz, size=TEST_SIZE)
			acc_computed = accuracy.eval(feed_dict={x: test_features[random_subarr, :, :, :],
						y_: np.reshape(test_labels[random_subarr, :, :], (TEST_SIZE, 15 * 15))})
			print("Accuracy:", acc_computed)
			print()

		generate_step += cur_multiplier
		if generate_step >= TRAV_LEN:
			generate_step = 0
			CUR_WH = 0
			CUR_BL = 0
			break

		saver.save(sess, './saved_sess_rollout_2/test_model', global_step=(epoch + 1))

saver.save(sess, './saved_sess_rollout_2/test_model_final')

#THE MODEL SAVES IN FOLDER NAME saved_sess_rollout_2
