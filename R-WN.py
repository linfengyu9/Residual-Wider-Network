
'''
We train and test Residual-Wider Network(R-WN) on the open datasets MNIST.
The source codes are as follows.
'''
import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
import time

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Hyperparameter
init_learning_rate = 1e-4
epsilon = 1e-8
dropout_rate = 0.2
nesterov_momentum = 0.9
weight_decay = 1e-4
class_num = 10
batch_size = 128
total_epochs = 50

def dropout(inputs, keep_prob, name):

    return tf.nn.dropout(inputs, keep_prob = keep_prob, name = name)

def _start_block(color_inputs):

    outputs = _conv2d(color_inputs, 7, 64, 2, name='conv1')
    outputs = _batch_norm(outputs, name='conv1', is_training=False, activation_fn=tf.nn.relu)
    outputs = _max_pool2d(outputs, 3, 2, name='pool1')
    return outputs

def _bottleneck_resblock(x, num_o, name, half_size=False, identity_connection=True):

    first_s = 2 if half_size else 1
    assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
    # branch 1
    if not identity_connection:
        o_b1 = _conv2d(x, 3, num_o, first_s, name='%s/bottleneck_v1/shortcut' % name)
        o_b1 = _batch_norm(o_b1, name='%s/bottleneck_v1/shortcut0' % name, is_training=False, activation_fn=None)
    else:
        o_b1 = x
    # branch 2
    o_b2b = _conv2d(x, 1, num_o/4, 1, name='%s/bottleneck_v1/conv2' % name)
    o_b2b = _batch_norm(o_b2b, name='%s/bottleneck_v1/conv22' % name, is_training=False, activation_fn=tf.nn.relu)

    o_b2e = _conv2d(o_b2b, 3, num_o/4, 1, name='%s/bottleneck_v1/conv4' % name)
    o_b2e = _batch_norm(o_b2e, name='%s/bottleneck_v1/conv44' % name, is_training=False, activation_fn=tf.nn.relu)
    # add
    o_b2badd1 = _add([o_b2b, o_b2e], name='addob2')
    o_b2c = _conv2d(o_b2badd1, 3, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
    o_b2c = _batch_norm(o_b2c, name='%s/bottleneck_v1/conv33' % name, is_training=False, activation_fn=None)
    # add
    outputs = _add([o_b1,o_b2c], name='%s/bottleneck_v1/add' % name)
    # relu
    outputs = _relu(outputs, name='%s/bottleneck_v1/relu' % name)
    return outputs

def _conv2d(x, kernel_size, num_o, stride, name, biased=False):

    channel_axis = 3
    num_x = x.shape[channel_axis].value
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        s = [1, stride, stride, 1]
        o = tf.nn.conv2d(x, w, s, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o

def _relu(x, name):

    return tf.nn.relu(x, name=name)

def _add(x_l, name):

    return tf.add_n(x_l, name=name)

def _max_pool2d(x, kernel_size, stride, name):

    k = [1, kernel_size, kernel_size, 1]
    s = [1, stride, stride, 1]
    return tf.nn.max_pool(x, k, s, padding='SAME', name=name)

def _batch_norm(x, name, is_training, activation_fn, trainable=False):

    with tf.variable_scope(name + '/BatchNorm') as scope:
        o = tf.contrib.layers.batch_norm(
            x,
            scale=True,
            activation_fn=activation_fn,
            is_training=is_training,
            trainable=trainable,
            scope=scope)
        return o

def shape(x):

    return str(x.get_shape())

def Global_Average_Pooling(x):

    return global_avg_pool(x, name='Global_avg_pooling')

def Linear(x) :

    return tf.layers.dense(inputs=x, units=10, name='linear')

def build(color_inputs):
    dropout_keep_prob = tf.where(True,0.2, 1.0)
    outputs = _start_block(color_inputs)
    print("after start block:", outputs.shape)

    # block 1
    outputs10 = _bottleneck_resblock(outputs, 256, 'unit_10', identity_connection=False)
    outputs11 = _bottleneck_resblock(outputs10, 256, 'unit_11', identity_connection=False)
    # add
    outputsb = _add([outputs10, outputs11], name='add1011')
    outputs12 = _bottleneck_resblock(outputsb, 256, 'unit_12', identity_connection=False)
    outputs = dropout(outputs12,dropout_keep_prob,'drop1')
    print("after block1:", outputs.shape)

    # block 2
    outputs20 = _bottleneck_resblock(outputs, 512, 'unit_20', identity_connection=False)
    outputs21 = _bottleneck_resblock(outputs20, 512, 'unit_21', identity_connection=False)
    # add
    outputsd = _add([outputs20,outputs21], name='add2021')
    outputs22 = _bottleneck_resblock(outputsd, 512, 'unit_22', identity_connection=False)
    print("after block2:", outputs22.shape)

    # block 3
    outputs30 = _bottleneck_resblock(outputs22, 1024, 'unit_30', identity_connection=False)
    outputs31 = _bottleneck_resblock(outputs30, 1024, 'unit_31', identity_connection=False)
    outputs32 = _bottleneck_resblock(outputs31, 1024, 'unit_32', identity_connection=False)
    outputs = dropout(outputs32, dropout_keep_prob, 'drop3')
    print("after block3:", outputs.shape)

    # Block 4
    x = Global_Average_Pooling(outputs)
    print('global_average_pooling shape is:', shape(x))

    x = flatten(x)
    print('flatten shape is:', shape(x))

    x = Linear(x)
    print('linear shape is:', shape(x))

    return x

x = tf.placeholder(tf.float32, shape=[None, 784])
batch_images = tf.reshape(x, [-1, 28, 28, 1])

label = tf.placeholder(tf.float32, shape=[None, 10])
training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = build(color_inputs=batch_images)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
train = optimizer.minimize(cost)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./R-WNmodel')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./R-WMlogs', sess.graph)

    global_step = 0
    epoch_learning_rate = init_learning_rate
    start_time = time.time()
    for epoch in range(total_epochs):
        if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        total_batch = int(mnist.train.num_examples / batch_size)

        for step in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }

            _, loss = sess.run([train, cost], feed_dict=train_feed_dict)

            if step % 100 == 0:
                global_step += 100
                train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)
                print("Step:", step, "Train Loss:", loss, "Training accuracy:", train_accuracy)
                writer.add_summary(train_summary, global_step=epoch)

            test_feed_dict = {
                x: mnist.test.images,
                label: mnist.test.labels,
                learning_rate: epoch_learning_rate,
                training_flag: False
            }

        accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)
        print('Epoch:', '%04d' % (epoch + 1), '/ Test Accuracy =', accuracy_rates)

    saver.save(sess=sess, save_path='./R-WNmodel/R-WN.ckpt')
    duration = time.time() - start_time
    print(duration)
