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
    trial = ai.Trial(output_path, clean=True)

    model = Model(**model_kw).init().to(device)
    opt = ai.opt.AdamW(model, lr=lr)

    val_ds, train_ds = ai.data.mnist_dataset().split(.1, .9)
    train_iter = train_ds.iterator(batch_size, device, train=True)
    val_iter = val_ds.iterator(256, device, train=False)

    task = ai.task.Classify(val_iter)

    trainer = ai.Trainer(ai.train.Classify(), train_iter)
    trainer.train(model, opt, trial.hook(True), steplimit=steps)

    print(task(model))


if __name__ == '__main__':
    ai.run(mnist)
