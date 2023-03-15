import ai
from ai.examples.mnist.model import Model


def mnist(
    output_path,
    device='cuda',
    batch_size=64,
    lr=1e-3,
    steps=5000,
    **model_kw,
):
    # data
    dataset = ai.data.mnist()
    val_ds, train_ds = dataset.split(.1, .9)
    val_iter = val_ds.iterator(256, device, train=False)
    train_iter = train_ds.iterator(batch_size, device, train=True)

    # task and trainer
    task = ai.task.Classify(val_iter)
    trainer = ai.Trainer(ai.train.Classify(), train_iter)

    # model and optimizer
    model = Model(**model_kw).init().to(device)
    opt = ai.opt.AdamW(model, lr=lr)

    # run a trial
    trial = ai.Trial(output_path, clean=True, task=task, val_data=val_iter)
    trainer.train(model, opt, trial.hook(), steplimit=steps)


if __name__ == '__main__':
    ai.run(mnist)
