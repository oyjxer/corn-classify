from pylab import *
from corn_net import corn_net
from solver import solver
from run_solvers import run_solvers
from eval_corn_net import eval_corn_net
from create_data import create_data
import numpy as np
import caffe, time


def train(niter, weights):
    start = time.time()

    # Reset style_solver as before.
    corn_solver_filename = solver(corn_net(train=True))
    corn_solver = caffe.get_solver(corn_solver_filename)
    corn_solver.net.copy_from(weights)

    # For reference, we also create a solver that isn't initialized from
    # the pretrained ImageNet weights.
    scratch_corn_solver_filename = solver(corn_net(train=True))
    scratch_corn_solver = caffe.get_solver(scratch_corn_solver_filename)

    print 'Running solvers for %d iterations...' % niter
    solvers = [('pretrained', corn_solver), ('scratch', scratch_corn_solver)]
    loss, acc, weights = run_solvers(niter, solvers)
    print 'Done.'

    end = time.time()
    print 'The stage of training use time: %.2fs' % (end - start)

    train_loss, scratch_train_loss = loss['pretrained'], loss['scratch']
    train_acc, scratch_train_acc = acc['pretrained'], acc['scratch']
    corn_weights, scratch_corn_weights = weights['pretrained'], weights['scratch']

    # Delete solvers to save memory.
    del corn_solver, scratch_corn_solver, solvers

    plot(np.vstack([train_loss, scratch_train_loss]).T)
    xlabel('Iteration #')
    ylabel('Loss')
    show()

    plot(np.vstack([train_acc, scratch_train_acc]).T)
    xlabel('Iteration #')
    ylabel('Accuracy')
    show()

    return corn_weights, scratch_corn_weights


def test(corn_weights, scratch_corn_weights):
    start = time.time()

    test_net, accuracy = eval_corn_net(corn_weights)
    print 'Accuracy, trained from ImageNet initialization: %3.1f%%' % (100 * accuracy,)
    scratch_test_net, scratch_accuracy = eval_corn_net(scratch_corn_weights)
    print 'Accuracy, trained from random initialization: %3.1f%%' % (100 * scratch_accuracy,)

    end = time.time()
    print 'The stage of testing use time: %.2fs' % (end - start)


if __name__ == '__main__':
    # set caffe
    caffe.set_mode_cpu()
    # caffe.set_mode_gpu()
    # caffe.set_device(0)

    data_path = raw_input("Please input image path: ")
    create_data(data_path)

    weights = 'model/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    niter = raw_input("Please input iterations in training: ")
    corn_weights, scratch_corn_weights = train(int(niter), weights)

    test(corn_weights, scratch_corn_weights)
