import numpy as np
import tensorflow as tf
import dataset_util


def softmax_classifier(x_tr, y_tr, x_te, y_te):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 3072], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')

    w = tf.Variable(tf.zeros([3072, 10]), name='W')
    tf.summary.histogram('W', w)
    b = tf.Variable(tf.zeros([10]), name='b')
    tf.summary.histogram('b', b)
    with tf.name_scope('Wx_b'):
        y = tf.nn.softmax(tf.matmul(x, w) + b)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(0.00001).minimize(cross_entropy)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.train.SummaryWriter("/tmp/cifar10/", sess.graph)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    batch_num = 0
    x_batches = []
    y_batches = []

    for i in np.array_split(x_tr, 5):
        x_batches.append(i)
    for i in np.array_split(y_tr, 5):
        y_batches.append(i)

    for i in range(1000):
        if batch_num == 5:
            batch_num = 0
        batch_xs = x_batches[batch_num]
        batch_ys = y_batches[batch_num]
        if ((i + 1) % 10) == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step],
                                  feed_dict={x: batch_xs, y_: batch_ys},
                                  options=run_options,
                                  run_metadata=run_metadata)
            writer.add_run_metadata(run_metadata, 'step%d' % (i + 1))
            writer.add_summary(summary, i + 1)
            if (i + 1) % 100 == 0:
                print "Train accuracy after step", i + 1, "\t:\t",
                print sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}) * 100, "%"
        else:
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        batch_num += 1

    print "Test accuracy\t:\t", (sess.run(accuracy, feed_dict={x: x_te, y_: y_te})) * 100, "%"


x_tr, y_tr, x_te, y_te = dataset_util.load_CIFAR10('dataset_pickle')

x_tr = np.reshape(x_tr, (x_tr.shape[0], -1))
x_te = np.reshape(x_te, (x_te.shape[0], -1))

one_hot_y_tr = []
one_hot_y_te = []
for i in y_tr:
    t = np.zeros(10)
    t[i] = 1
    one_hot_y_tr.append(t)

for i in y_te:
    t = np.zeros(10)
    t[i] = 1
    one_hot_y_te.append(t)
softmax_classifier(x_tr, one_hot_y_tr, x_te, one_hot_y_te)
