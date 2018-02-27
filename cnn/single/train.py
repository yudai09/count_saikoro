import chainer
from chainer import links as L
from chainer import training
from chainer.training import extensions
from gen_image_dataset import gen_image_dataset

from models.CNN import CNN

def main(batchsize=10, gpu=-1, epoch = 100, resume = False):
    train, test = gen_image_dataset()

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    # test_iter = chainer.iterators.SerialIterator(test, batchsize,
    #                                              repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, 1,
                                                 repeat=False, shuffle=False)

    model = L.Classifier(CNN())

    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu()

    opt = chainer.optimizers.Adam()
    opt.setup(model)

    updater = training.StandardUpdater(train_iter, opt, device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='./result')

    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'snapshot_{.updater.iteration}'))

    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    trainer.extend(extensions.ProgressBar())

    if resume:
        chainer.serializers.load_npz(resume, trainer)

    trainer.run()

    chainer.serializers.save_npz('./saikoro_model.npz', model)


if __name__ == '__main__':
    main()
