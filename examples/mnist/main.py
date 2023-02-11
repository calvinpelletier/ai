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
    opt = ai.opt.adam(model, lr=lr)

    val_ds, train_ds = ai.data.mnist_dataset().split(.1, .9)
    train_loader = train_ds.loader(batch_size, device, train=True)
    val_loader = val_ds.loader(256, device, train=False)

    task = ai.task.Classify(val_loader)

    trainer = ai.Trainer(ai.train.Classify(), train_loader)
    trainer.train(model, opt, trial.hook(True), steplimit=steps)

    print(task(model))


if __name__ == '__main__':
    ai.run(mnist)
