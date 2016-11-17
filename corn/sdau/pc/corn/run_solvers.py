import numpy as np
import tempfile, os


def run_solvers(niter, solvers, disp_interval=10):
    """Run solvers for niter iterations, returning the loss and accuracy recorded each iteration. `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers} for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy() for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join(
                '%s: loss=%.3f, acc=%2d%%' % (n, loss[n][it], np.round(100 * acc[n][it])) for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)
    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights
