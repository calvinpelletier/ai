import ai
from ai.examples.vit.model import VisionTransformer


def vit_cifar10(
    output_path,
    device='cuda',
    batch_size=64,
    lr=1e-3,
    steps=5000,
    **model_kw,
):
    val_ds, train_ds = ai.data.cifar10().split(.1, .9)
    train_iter = train_ds.iterator(batch_size, device, train=True)
    val_iter = val_ds.iterator(256, device, train=False)

    trial = ai.Trial(
        output_path,
        val_data=val_iter,
        task=ai.task.Classify(val_iter),
        clean=True,
    )

    model = VisionTransformer(
        imsize=32,
        patch_size=8,
        n_out=10,
        dim=64,
        n_blocks=2,
        n_heads=2,
        mlp_dim=64,
    ).init().to(device)

    ai.Trainer(ai.train.Classify(), train_iter).train(
        model,
        ai.opt.AdamW(model, lr=lr),
        trial.hook(),
        steplimit=steps,
    )


if __name__ == '__main__':
    ai.run(vit_cifar10)
