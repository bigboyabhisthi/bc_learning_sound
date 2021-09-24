"""
 Learning from Between-class Examples for Deep Sound Recognition.
 Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada

"""

import sys
import os
import chainer
import json

import opts
import models
import dataset
from train import Trainer

from pathlib import Path


def main():
    opt = opts.parse()
    chainer.cuda.get_device_from_id(opt.gpu).use()
    for split in opt.splits:
        print("+-- Split {} --+".format(split))
        train(opt, split)


def train(opt, split):

    start_epoch = 0
    model = getattr(models, opt.netType)(opt.nClasses)
    model_root_dir = Path(opt.save)
    if "model_split_{}.npz".format(split) in [f for f in os.listdir(model_root_dir)]:
        print("Pre-trained available for model_split_{}.npz".format(split))
        chainer.serializers.load_npz(
            model_root_dir / "model_split_{}.npz".format(split), model
        )
        with open(model_root_dir / "epoch_{}.json".format(split), "r") as fp:
            start_epoch = json.load(fp)["epoch"]

    print("Starting training at epoch: {}".format(start_epoch + 1))

    model.to_gpu()
    optimizer = chainer.optimizers.NesterovAG(lr=opt.LR, momentum=opt.momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(opt.weightDecay))
    train_iter, val_iter = dataset.setup(opt, split)
    trainer = Trainer(model, optimizer, train_iter, val_iter, opt)

    if opt.testOnly:
        chainer.serializers.load_npz(
            os.path.join(opt.save, "model_split{}.npz".format(split)), trainer.model
        )
        val_top1 = trainer.val()
        print("| Val: top1 {:.2f}".format(val_top1))
        return

    for epoch in range(start_epoch + 1, opt.nEpochs + 1):
        train_loss, train_top1 = trainer.train(epoch)
        val_top1 = trainer.val()
        sys.stderr.write("\r\033[K")
        sys.stdout.write(
            "| Epoch: {}/{} | Train: LR {}  Loss {:.3f}  top1 {:.2f} | Val: top1 {:.2f}\n".format(
                epoch,
                opt.nEpochs,
                trainer.optimizer.lr,
                train_loss,
                train_top1,
                val_top1,
            )
        )
        if epoch % 50 == 0:
            chainer.serializers.save_npz(
                model_root_dir / "model_split_{}.npz".format(split), model
            )
            with open(model_root_dir / "epoch_{}.json".format(split), "w") as fp:
                json.dump({"epoch": epoch}, fp)
        sys.stdout.flush()

    if opt.save != "None":
        chainer.serializers.save_npz(
            os.path.join(opt.save, "model_split{}.npz".format(split)), model
        )


if __name__ == "__main__":
    main()
