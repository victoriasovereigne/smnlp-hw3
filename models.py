# models.py

import tensorflow as tf
import numpy as np
import random
from sentiment_data import *


# Returns a new numpy array with the data from np_arr padded to be of length length. If length is less than the
# length of the base array, truncates instead.
def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result


# Train a feedforward neural network on the given training examples, using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value). Returned predictions
# should be SentimentExample objects with predicted labels and the same sentences as input (but these won't be
# read for evaluation anyway)

# train_exs is a list of SentimentExample objects
# word_vectors is a list of WordEmbeddings objects
def train_ffnn(train_exs, dev_exs, test_exs, word_vectors):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])

    # development and test data
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    test_labels_arr = np.array([ex.label for ex in test_exs])

    word_indexer = word_vectors.word_indexer # word indexer
    vectors = word_vectors.vectors # the word embeddings 

    # =========================================================================
    # Construct the training data
    # =========================================================================
    train_xs = []

    for sent_num in xrange(len(train_exs)):
        tmp = np.zeros(shape=(len(vectors[0])))

        for word_num in xrange(train_seq_lens[sent_num]):
            index = int(train_mat[sent_num, word_num])
            embeddings = vectors[index]
            tmp += embeddings

        tmp /= train_seq_lens[sent_num] # divide by sentence length like in slide
        train_xs.append(tmp)
    
    train_xs = np.array(train_xs)
    train_ys =  train_labels_arr
    
    # =========================================================================
    # Construct the development data
    # =========================================================================
    dev_xs = []

    for sent_num in xrange(len(dev_exs)):
        tmp = np.zeros(shape=(len(vectors[0])))

        for word_num in xrange(dev_seq_lens[sent_num]):
            index = int(dev_mat[sent_num, word_num])
            embeddings = vectors[index]
            tmp += embeddings

        tmp /= dev_seq_lens[sent_num] # divide by sentence length like in slide
        dev_xs.append(tmp)
    
    dev_xs = np.array(dev_xs)
    dev_ys =  dev_labels_arr

    # =========================================================================
    # Construct the test data
    # =========================================================================
    test_xs = []

    for sent_num in xrange(len(test_exs)):
        tmp = np.zeros(shape=(len(vectors[0])))

        for word_num in xrange(test_seq_lens[sent_num]):
            index = int(test_mat[sent_num, word_num])
            embeddings = vectors[index]
            tmp += embeddings

        tmp /= test_seq_lens[sent_num] # divide by sentence length like in slide
        test_xs.append(tmp)
    
    test_xs = np.array(test_xs)
    test_ys =  test_labels_arr

    # =========================================================================
    # Start the FFNN
    # =========================================================================
    feat_vec_size = len(vectors[0])
    hidden_size = 10 
    num_classes = 2

    fx = tf.placeholder(tf.float32, feat_vec_size) # vector length 50
    # fx = tf.placeholder(tf.float32, [batch_size, feat_vec_size])

    # 10 x 50 matrix
    V = tf.get_variable("V", [hidden_size, feat_vec_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    # 10 x 1 matrix
    z = tf.sigmoid(tf.tensordot(V, fx, 1))
    W1 = tf.get_variable("W1", [hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # W2 = tf.get_variable("W2", [hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3", [num_classes, hidden_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    # h1 = tf.nn.softmax(tf.tensordot(W1, z, 1))
    h1 = tf.nn.sigmoid(tf.tensordot(W1, z, 1))
    # keep_prob = tf.placeholder(tf.float32) 
    # h2 = tf.nn.softmax(tf.tensordot(W2, h1, 1))
    probs = tf.nn.softmax(tf.tensordot(W3, h1, 1))
    # dropout = tf.nn.dropout(h2, keep_prob)
    # probs = tf.nn.softmax(tf.tensordot(W3, dropout, 1))

    # probs = tf.nn.softmax(tf.tensordot(W, z, 1))
    label = tf.placeholder(tf.int32, 1)

    one_best = tf.argmax(probs)
    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes])
    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))

    decay_steps = 1000
    learning_rate_decay_factor = 0.99
    global_step = tf.contrib.framework.get_or_create_global_step()
    initial_learning_rate = 0.1

    lr = tf.train.exponential_decay(initial_learning_rate, 
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    
    # Tensorboard thingy
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)

    # opt = tf.train.AdamOptimizer(lr)
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # Training and testing
    init = tf.global_variables_initializer()
    num_epochs = 200
    merged = tf.summary.merge_all()
    batch_num = 0


    # compute the loss
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        tf.set_random_seed(0)
        sess.run(init)

        step_idx = 0

        for i in range(0, num_epochs):
            loss_this_iter = 0

            for ex_idx in xrange(0, len(train_seq_lens)):
                [_, loss_this_instance, summary, probs_this_instance, pred_this_instance, z_this_instance] = sess.run([apply_gradient_op, loss, merged, probs, one_best, z], 
                                                            feed_dict={fx:train_xs[ex_idx],
                                                                        label:np.array([train_ys[ex_idx]]),
                                                                        # keep_prob:0.75
                                                                        })

                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)


        # =========================================================================
        # Evaluate on the train set
        # =========================================================================
        train_correct = 0
        predicted = []

        for ex_idx in xrange(0, len(train_seq_lens)):
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                    feed_dict={fx: train_xs[ex_idx],
                                                                                                # keep_prob:1.0
                                                                                                })

            pred_sentiment = SentimentExample(train_exs[ex_idx].indexed_words, pred_this_instance)
            predicted.append(pred_sentiment)
            if (train_ys[ex_idx] == pred_this_instance):
                train_correct += 1

        print repr(train_correct) + "/" + repr(len(train_labels_arr)) + " correct after training"
        print float(train_correct) / len(train_labels_arr)

        # =========================================================================
        # Evaluate on the dev set
        # =========================================================================
        dev_correct = 0
        predicted_dev = []

        for ex_idx in xrange(0, len(dev_seq_lens)):
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                    feed_dict={fx: dev_xs[ex_idx],
                                                                                                # keep_prob:1.0
                                                                                                })
            
            pred_sentiment = SentimentExample(dev_exs[ex_idx].indexed_words, pred_this_instance)
            predicted_dev.append(pred_sentiment)
            if (dev_ys[ex_idx] == pred_this_instance):
                dev_correct += 1

        print repr(dev_correct) + "/" + repr(len(dev_labels_arr)) + " correct after development"
        print float(dev_correct) / len(dev_labels_arr)

        # =========================================================================
        # Predict on the test set
        # =========================================================================
        predicted_test = []

        for ex_idx in xrange(0, len(test_seq_lens)):
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                    feed_dict={fx: test_xs[ex_idx],
                                                                                                # keep_prob:1.0
                                                                                                })
            
            pred_sentiment = SentimentExample(test_exs[ex_idx].indexed_words, pred_this_instance)
            predicted_test.append(pred_sentiment)

    return predicted_test


# Analogous to train_ffnn, but trains your fancier model.
# Ref: https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow
def train_fancy(train_exs, dev_exs, test_exs, word_vectors):
    seq_max_len = 60
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    train_labels_arr = np.array([ex.label for ex in train_exs])

    word_indexer = word_vectors.word_indexer # word indexer
    vectors = word_vectors.vectors # the word embeddings 

    # =========================================================================
    # Construct the training data
    # =========================================================================   
    train_xs = np.array(train_mat)
    train_ys = [] # train_labels_arr

    for element in train_labels_arr:
        ans = [0, 0]
        ans[element] = 1
        train_ys.append(ans)

    train_ys = np.array(train_ys)

    # =========================================================================
    # Construct the dev data
    # =========================================================================
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    dev_xs = np.array(dev_mat)
    dev_ys = [] # dev_labels_arr

    for element in dev_labels_arr:
        ans = [0, 0]
        ans[element] = 1
        dev_ys.append(ans)

    dev_ys = np.array(dev_ys)

    # =========================================================================
    # Construct the test data
    # =========================================================================
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    test_labels_arr = np.array([ex.label for ex in test_exs])

    test_xs = np.array(test_mat)
    test_ys = [] # test_labels_arr

    for element in test_labels_arr:
        ans = [0, 0]
        ans[element] = 1
        test_ys.append(ans)

    test_ys = np.array(test_ys)

    # =========================================================================
    # Create batches
    # =========================================================================
    batch_size = 10
    num_classes = 2
    lstm_size = 64 #128 #64
    feat_vec_size = len(vectors[0])

    tf.reset_default_graph()
    labels = tf.placeholder(tf.float32, [None, num_classes])
    
    input_data = tf.placeholder(tf.int32, [None, seq_max_len])
    data = tf.nn.embedding_lookup(vectors.astype('float32'), input_data)

    mode = 'basic' #bidirectional

    # basic LSTM 
    if mode == 'basic':
        print "basic"
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        # lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
        
        value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

        # weight = tf.Variable(tf.truncated_normal([lstm_size, num_classes]))
        weight = tf.get_variable("W", [lstm_size, num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        # bias = tf.get_variable("b", [num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        
        value = tf.transpose(value, [1,0,2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.tensordot(last, weight, 1) + bias)

    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
    elif mode == 'bidirectional':
        print "bidirectional"
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        # lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=0.75)

        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        # lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, output_keep_prob=0.75)
        
        data = tf.unstack(data, seq_max_len, 1)
        value, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, data, dtype=tf.float32)

        weight = tf.get_variable("W", [2*lstm_size, num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

        prediction = tf.matmul(value[-1], weight) + bias  

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # one_best = tf.argmax(prediction)

    decay_steps = 1000
    learning_rate_decay_factor = 0.99
    global_step = tf.contrib.framework.get_or_create_global_step()
    initial_learning_rate = 0.001

    lr = tf.train.exponential_decay(initial_learning_rate, 
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer(lr)
    # optimizer = tf.train.GradientDescentOptimizer(lr)
    grads = optimizer.compute_gradients(loss)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')
    
    init = tf.global_variables_initializer()
    num_epochs = 100

    with tf.Session() as sess:
        sess.run(init)

        for i in range(num_epochs):
            loss_this_iter = 0
            batch_num = 0

            while batch_num < len(train_seq_lens):
                train_xs_batch = train_xs[batch_num:batch_num+batch_size]
                train_ys_batch = train_ys[batch_num:batch_num+batch_size]

                [_, loss_this_instance] = sess.run([train_op, loss],
                                                    feed_dict={input_data:train_xs_batch, 
                                                                labels:train_ys_batch})

                loss_this_iter += loss_this_instance
                batch_num += batch_size

            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)

            if loss_this_iter < 0.1:
                break

        # evaluation
        correct = sess.run(accuracy, {input_data: train_xs, labels:train_ys})
        print "train --> correct:", correct

        correct = sess.run(accuracy, {input_data: dev_xs, labels:dev_ys})
        print "dev --> correct:", correct

        # dev evaluation
        batch_num = 0
        index = 0
        predicted_dev = []
        total_correct = 0

        while batch_num < len(dev_seq_lens):
            dev_xs_batch = dev_xs[batch_num:batch_num+batch_size]
            pred = sess.run(prediction, feed_dict={input_data:dev_xs})

            for values in pred:
                if index < len(dev_seq_lens):
                    true_y = dev_labels_arr[index]
                    pred_y = np.argmax(values)
                    pred_sentiment = SentimentExample(dev_exs[index].indexed_words, pred_y)
                    predicted_dev.append(pred_sentiment)

                    if true_y == pred_y:
                        total_correct += 1
                    index += 1
            batch_num += batch_size

        print total_correct / float(len(dev_seq_lens))
        
        # test evaluation
        batch_num = 0
        index = 0
        predicted_test = []
        total_correct = 0

        while batch_num < len(test_seq_lens):
            test_xs_batch = test_xs[batch_num:batch_num+batch_size]
            pred = sess.run(prediction, feed_dict={input_data:test_xs})

            for values in pred:
                if index < len(test_seq_lens):
                    pred_y = np.argmax(values)
                    pred_sentiment = SentimentExample(test_exs[index].indexed_words, pred_y)
                    predicted_test.append(pred_sentiment)
                    index += 1
            batch_num += batch_size

    return predicted_test

    # python /v/filer5b/v20q001/vlestari/.local/lib/python2.7/site-packages/tensorboard/main.py
