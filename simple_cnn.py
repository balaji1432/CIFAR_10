import numpy as np
import dataset_util
import tensorflow as tf


def simple_cnn(x_tr, y_tr, x_te, y_te):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x')
        y_ = tf.placeholder(tf.float32, [None, 10], name='x')

    with tf.name_scope('layer1'):
        W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.1), name='W_conv1')
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv1')
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, [1, 1, 1, 1], padding='SAME') + b_conv1, name='h_conv1')
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')
        tf.summary.histogram('W_conv1', W_conv1)
        tf.summary.histogram('b_conv1', b_conv1)
        with tf.variable_scope('visualization'):
            x_min = tf.reduce_min(W_conv1)
            x_max = tf.reduce_max(W_conv1)
            kernel_0_to_1 = (W_conv1 - x_min) / (x_max - x_min)
            kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
            tf.image_summary('conv1/filters', kernel_transposed, max_images=10)

    with tf.name_scope('layer2'):
        W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1), name='W_conv2')
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv2')
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, [1, 1, 1, 1], padding='SAME') + b_conv2, name='h_conv2')
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')
        tf.summary.histogram('W_conv2', W_conv2)
        tf.summary.histogram('b_conv2', b_conv2)

    with tf.name_scope('layer3'):
        W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), name='W_conv3')
        b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]), name='b_conv3')
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, [1, 1, 1, 1], padding='SAME') + b_conv3, name='h_conv3')
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool3')
        tf.summary.histogram('W_conv3', W_conv3)
        tf.summary.histogram('b_conv3', b_conv3)

    with tf.name_scope('fully_conected'):
        W_fc1 = tf.Variable(tf.truncated_normal([4 * 4 * 128, 1024], stddev=0.1), name='W_fc1')
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b_fc1')
        h_pool2_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 128], name='h_pool2_flat')
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='h_fc1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_dropout')

    with tf.name_scope('readout'):
        W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name='W_fc2')
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]), name='b_fc2')
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    with tf.name_scope('train'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_), name='cross_entropy')
        train_step = tf.train.AdamOptimizer(1e-3, name='train_step').minimize(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy)

    sess = tf.Session()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.train.SummaryWriter("/Users/Balaji/Desktop/cifar_10_conv/logs/", sess.graph)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    batch_num = 0
    x_batches = []
    y_batches = []

    """
    t = np.array_split(x_tr, 5)
    for i in np.array_split(t[0], 5):
        x_batches.append(i)
    t = np.array_split(y_tr, 5)
    for i in np.array_split(t[0], 5):
        y_batches.append(i)
    """

    for i in np.array_split(x_tr, 5):
        x_batches.append(i)
    for i in np.array_split(y_tr, 5):
        y_batches.append(i)

    for i in range(30):
        if batch_num == 5:
            batch_num = 0
        batch_xs = x_batches[batch_num]
        batch_ys = y_batches[batch_num]
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0}, options=run_options,
                           run_metadata=run_metadata)
        writer.add_run_metadata(run_metadata, 'step%d' % (i + 1))
        writer.add_summary(summary, i + 1)
        if (i + 1) % 10 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i + 1, train_accuracy * 100))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    print("test accuracy %g" % (sess.run(accuracy, feed_dict={x: x_te, y_: y_te, keep_prob: 1.0}) * 100))
    save_path = saver.save(sess, "/Users/Balaji/Desktop/cifar_10_conv/saved_weights/model.ckpt")
    print("Model saved in file: %s" % save_path)


x_tr, y_tr, x_te, y_te = dataset_util.load_CIFAR10('dataset_pickle')
one_hot_y_tr = []
one_hot_y_te = []
for i in y_tr:
    t = np.zeros(10)
    t[i] = 1
    one_hot_y_tr.append(t)

x_te = (np.array_split(x_te, 10))[0]
y_te = (np.array_split(y_te, 10))[0]
for i in y_te:
    t = np.zeros(10)
    t[i] = 1
    one_hot_y_te.append(t)

simple_cnn(x_tr, one_hot_y_tr, x_te, one_hot_y_te)
