# -*- coding:utf-8 -*-
# I have referenced below example of Chainer
# https://github.com/chainer/chainer/blob/master/examples/cifar/train_cifar.py

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

import models.VGG

from gen_image_dataset import gen_image_dataset

def main(batchsize=128, learnrate=0.1, epoch=300, gpu=-1, out='result', resume='', early_stopping=False):
    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batchsize))
    print('# epoch: {}'.format(epoch))
    print('')

    label_eyes, train, test = gen_image_dataset()
    class_labels = len(label_eyes)

    model = L.Classifier(models.VGG.VGG(class_labels))
    if gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.MomentumSGD(learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)

    stop_trigger = (epoch, 'epoch')
    # Early stopping option
    if early_stopping:
        stop_trigger = triggers.EarlyStoppingTrigger(
            monitor=early_stopping, verbose=True,
            max_trigger=(epoch, 'epoch'))

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, stop_trigger, out=out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
