import ai
from ai.examples.vit.model import VisionTransformer


def mnist_vit(
    output_path,
    device='cuda',
    batch_size=64,
    lr=1e-3,
    steps=5000,
    **model_kw,
):
    val_ds, train_ds = ai.data.mnist_dataset().split(.1, .9)
    train_loader = train_ds.loader(batch_size, device, train=True)
    val_loader = val_ds.loader(256, device, train=False)

    trial = ai.Trial(output_path, clean=True, val_data=val_loader)

    model = VisionTransformer(
        imsize=28,
        patch_size=7,
        n_out=10,
        dim=64,
        n_blocks=2,
        n_heads=2,
        head_dim=64,
        mlp_dim=64,
        nc_in=1,
        dropout=.1,
    ).init().to(device)
    opt = ai.opt.adam(model, lr=lr)

    trainer = ai.Trainer(ai.train.Classify(), train_loader)
    trainer.train(model, opt, trial.hook(), steplimit=steps)

    task = ai.task.Classify(val_loader) # TODO: task hook
    print(task(model))


if __name__ == '__main__':
    ai.run(mnist_vit)
