from a3c.a3c_v1 import *
from a3c.worker_v1 import *
import basenet.transfer as transfer


max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = (512, 512) # Observations are greyscale frames of 84 * 84 * 1
load_model = False
load_base = True
model_path = './model'
base_path = './basenet/unet_trained'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes',
                                  trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network(s_size, 'global', None, debug=False)  # Generate global network
    # num_workers = multiprocessing.cpu_count()  # Set workers to number of available CPU threads
    num_workers = 1
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(i, s_size, trainer, model_path,
                              global_episodes, 'cycle',
                              data_path='D:\\Datasets\\cycle'))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        # sess.run(tf.global_variables_initializer())
        w1 = tf.get_default_graph().get_tensor_by_name('global/unet/down_conv_0/w1:0')
        print_op = tf.Print(w1, [w1])
        inited = None
        if load_base:
            inited = transfer.param_transfer(sess, base_path, master_network.unet_var_dict)
            sess.run(print_op)
        transfer.guarantee_initialized_variables(sess, inited)
        sess.run(print_op)

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)