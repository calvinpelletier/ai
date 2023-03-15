import ai
from ai.model.ae import ImgAutoencoder
from ai.data.datasets.ffhq import MISSING_FFHQ_MSG


# study
STUDY = 'imsim' # path relative to $AI_LAB_PATH

# constants
DEVICE = 'cuda'
IMSIZE = 64
PERCEP_TYPE = 'lpips-alex'
NC_MIN = 16
NC_MAX = 256
LOSSES = ['pixel', 'percep', 'face_id', 'combo']


# dataset
try:
    ds = ai.data.ImgDataset('ffhq', IMSIZE)
except ValueError as e:
    print(MISSING_FFHQ_MSG.format(imsize=IMSIZE))
    raise e
val_ds, train_ds = ds.split(.01, .99)
val_iter = val_ds.iterator(128, DEVICE, train=False)
samples = val_ds.sample(8, DEVICE)

# model
model = ImgAutoencoder(IMSIZE, 4, NC_MIN, NC_MAX).to(DEVICE)


class CLI:
    def clean(s):
        '''Delete and recreate the study directory.'''

        ai.Study(STUDY, clean=True)

    def train(s, loss, hp=None, steplimit=10_000):
        '''Train the autoencoder using the specified loss function.'''

        study = ai.Study(STUDY)

        # create a trial inside the study
        trial = study.trial(
            loss,
            clean=True, # delete this trial if it already exists
            save_snapshots=True, # regularly save the model and optimizer
            val_data=val_iter, # regularly run validation

            # save side-by-side comparisons of sample inputs and their resulting
            # outputs at regular intervals during training
            sampler=lambda path, step, model: ai.util.save_img_grid(
                path / f'{step}.png',
                [samples, model(samples)],
            ),
        )

        if hp is None:
            # get the best hyperparameters from the search
            hp = study.experiment(f'hps/{loss}').best_hparams

        # run training
        run(
            build_loss(loss),
            trial.hook(),
            hp['batch_size'],
            hp['learning_rate'],
            hp['grad_clip'],
            steplimit=steplimit,
        )

    def hps(s, loss='all', n=16, steplimit=5000, prune=True, clean=False):
        '''Run a hyperparameter search.'''

        if loss == 'all':
            for loss in LOSSES:
                exp = run_hps(loss, n, steplimit, prune, clean)
                print(loss, exp.best_hparams)
        else:
            exp = run_hps(loss, n, steplimit, prune, clean)
            print(exp.best_hparams)

    def compare(s):
        '''Create comparison image of all the different loss functions.'''

        study = ai.Study(STUDY)
        model.eval()
        comparison = [samples] # list of image batches
        for loss in LOSSES:
            model.init(study.trial(loss).model_path()) # load params from disk
            comparison.append(model(samples))
        ai.util.save_img_grid(study.path / 'comparison.png', comparison)

    def plot_hparam(s, loss, hparam):
        study = ai.Study(STUDY)
        exp = study.experiment(f'hps/{loss}')
        print(exp.best_hparams)
        exp.show_plot(hparam)


def run(loss_fn, hook, batch_size, lr, grad_clip, steplimit=5000):
    trainer = ai.Trainer(
        ai.train.Reconstruct(loss_fn), # training environment
        train_ds.iterator(batch_size, DEVICE, train=True), # training data
    )

    trainer.train(
        model.init(),
        ai.opt.AdamW(model, lr=lr, grad_clip=grad_clip),
        hook,
        steplimit=steplimit,
    )

    return trainer.validate(model, val_iter)


def run_hps(loss, n, steplimit, prune, clean):
    # create an experiment inside the study
    exp = ai.Study(STUDY).experiment(
        f'hps/{loss}',
        clean=clean,
        val_data=val_iter,
        prune=prune,
    )

    exp.run(n, lambda trial: run(
        build_loss(loss),

        # handles validation and early stopping to prune unpromising trials
        trial.hook(),

        # both specifies the searchable hyperparameter space for the whole
        # experiment and selects the exact hparams for this specific trial.
        trial.hp.lin('batch_size', 8, 32, step=8), # linear
        trial.hp.log('learning_rate', 1e-4, 1e-2), # logarithmic
        trial.hp.lst('grad_clip', [False, True]), # list

        steplimit=steplimit,
    ))

    return exp


def build_loss(name):
    if name == 'pixel':
        return ai.loss.L2Loss()
    elif name == 'percep':
        return ai.loss.PerceptualLoss(PERCEP_TYPE)
    elif name == 'face_id':
        return ai.loss.FaceIdentityLoss()
    elif name == 'combo':
        return ai.loss.ComboLoss(
            pixel=ai.loss.L2Loss(),
            percep=ai.loss.PerceptualLoss(PERCEP_TYPE),
            face_id=(ai.loss.FaceIdentityLoss(), 0.1), # weight: 0.1
        )
    raise ValueError(name)


if __name__ == '__main__':
    ai.run(CLI)
