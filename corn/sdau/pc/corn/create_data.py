import os
import random


def create_data(data_path, thres=0.7):
    label = -1

    test_file = 'data/corn/test.txt'
    train_file = 'data/corn/train.txt'

    test = open(test_file, 'w')
    train = open(train_file, 'w')

    for dir in os.listdir(data_path):
        if os.path.isfile(os.path.join(data_path, dir)):
            continue
        label += 1
        for root, _, files in os.walk(os.path.join(data_path, dir)):
            for file in files:
                image_file = os.path.join(root, file)
                if random.random() > thres:
                    test.write(os.path.abspath(image_file) + ' ' + str(label) + '\n')
                else:
                    train.write(os.path.abspath(image_file) + ' ' + str(label) + '\n')

    test.close()
    train.close()
