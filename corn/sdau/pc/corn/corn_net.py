from caffenet import caffenet
from caffe import layers as L

NUM_CORN_LABELS = 2


def corn_net(train=True, learn_all=False, subset=None):
    if subset is None:
        subset = 'train' if train else 'test'
    source = 'data/corn/%s.txt' % subset
    transform_param = dict(mirror=train, crop_size=227, mean_file='data/ilsvrc12/imagenet_mean.binaryproto')
    corn_data, corn_label = L.ImageData(transform_param=transform_param, source=source, batch_size=50, new_height=256,
                                        new_width=256, ntop=2)
    return caffenet(data=corn_data, label=corn_label, train=train, num_classes=NUM_CORN_LABELS,
                    classifier_name='fc8_corn', learn_all=learn_all)
