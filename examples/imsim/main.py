import ai
from ai.model.ae import ImgAutoencoder


study = ai.Study('imsim')

device = 'cuda'
imsize = 64
batch_size = 32
val_batch_size = 128
percep_type = 'lpips-alex'
nc_min = 16
nc_max = 256

val_ds, train_ds = ai.data.ImgDataset('ffhq', imsize).split(.01, .99)
samples = val_ds.sample(8, device)


def build_data_loaders():
    train_data = train_ds.loader(batch_size, device, train=True)
    val_data = val_ds.loader(val_batch_size, device, train=False)
    return train_data, val_data


def build_model(bottleneck):
    return ImgAutoencoder(imsize, bottleneck, nc_min, nc_max).to(device)


def build_loss(name):
    if name == 'pixel':
        return ai.loss.L2Loss()
    elif name == 'percep':
        return ai.loss.PerceptualLoss(percep_type)
    elif name == 'face_id':
        return ai.loss.FaceIdentityLoss()
    elif name == 'combo':
        return ai.loss.ComboLoss(
            ai.loss.L2Loss(),
            ai.loss.PerceptualLoss(percep_type),
            (ai.loss.FaceIdentityLoss(), 0.1),
        )
    raise ValueError(name)


def run(
    trial,
    train_data,
    val_data,
    loss,
    model,
    opt,
    steplimit=None,
):
    trainer = ai.Trainer(ai.train.Reconstruct(loss), train_data)
    return trainer.train(model.init(), opt, trial.hook(), steplimit=steplimit)


class CLI:
    def clean(s):
        study.clean()

    def train(s, loss='pixel', bottleneck=4, lr=4e-4, steps=None):
        train_data, val_data = build_data_loaders()

        trial = study.trial(
            f'{loss}/{bottleneck}/main',
            clean=True,
            save_snapshots=True,
            val_data=val_data,
            sampler=lambda path, step, model: ai.util.img.save_img_grid(
                path / f'{step}.png',
                [samples, model(samples)],
            ),
        )

        loss_fn = build_loss(loss)
        model = build_model(bottleneck)
        opt = ai.opt.adam(model, lr=lr)

        run(trial, train_data, val_data, loss_fn, model, opt, steps)

    def hypertrain(s, n=8, loss='pixel', bottleneck=4, steps=4000):
        train_data, val_data = build_data_loaders()

        exp = study.experiment(f'{loss}/{bottleneck}/hps', val_data=val_data)

        loss_fn = build_loss(loss)
        model = build_model(bottleneck)

        exp.run(n, lambda trial: run(
            trial,
            train_data,
            val_data,
            loss_fn,
            model,
            ai.opt.adam(model, lr=trial.hp.log('lr', 1e-4, 1e-2)),
            steps,
        ))

        print(exp.best_hparams())

    def best_hparams(s, loss='pixel', bottleneck=4):
        exp = study.experiment(f'{loss}/{bottleneck}/hps')
        print(exp.best_hparams())
        exp.show_plot('lr')

    def analyze(s):
        bottleneck = 4
        grid = [samples]
        for loss in ['pixel', 'percep', 'face_id', 'combo']:
            trial = study.trial(f'{loss}/{bottleneck}/main')
            model = build_model(bottleneck).init(trial.model_path()).eval()
            grid.append(model(samples))
        ai.util.img.save_img_grid(study.path / 'losses.png', grid)

        loss = 'pixel'
        grid = [samples]
        for bottleneck in [16, 8, 4]:
            trial = study.trial(f'{loss}/{bottleneck}/main')
            model = build_model(bottleneck).init(trial.model_path()).eval()
            grid.append(model(samples))
        ai.util.img.save_img_grid(study.path / 'bottlenecks.png', grid)


if __name__ == '__main__':
    ai.run(CLI)
