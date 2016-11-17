from corn_net import corn_net
import caffe


def eval_corn_net(weights, test_iters=10):
    test_net = caffe.Net(corn_net(train=False), weights, caffe.TEST)
    accuracy = 0
    for it in xrange(test_iters):
        accuracy += test_net.forward()['acc']
    accuracy /= test_iters
    return test_net, accuracy
