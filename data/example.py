import ai

batch_size = 64
device = 'cuda'

# load (download first if needed) the MNIST dataset from $AI_DATASETS_PATH/mnist
ds = ai.data.mnist_dataset()
# or alternatively, provide a path
ds = ai.data.mnist_dataset('/tmp/mnist')

# split into a train set and a validation set
train_ds, val_ds = ds.split() # standard split
train_ds, val_ds = ds.split(.9, .1) # custom split

# check length
print(len(train_ds)) # 63000
print(train_ds.length(batch_size)) # 62976 (the length accounting for dropping
                                   # the last incomplete batch)

# load and examine 100 samples
samples = val_ds.sample(100, device)
ai.util.print_info(samples['x']) # shape=[100,1,28,28] bounds=[-1.00,1.00]
                                 # dtype=float32 device=cuda:0
ai.util.print_info(samples['y']) # shape=[100] bounds=[0,9] dtype=uint8
                                 # device=cuda:0

# train loader (shuffles and loops infinitely)
train_loader = train_ds.loader(batch_size, device, train=True)
# val loader (doesnt shuffle and loops for one epoch)
val_loader = val_ds.loader(batch_size, device, train=False)

# iterate via loop:
for batch in val_loader:
    pass
# or:
val_iterator = iter(val_loader)
batch = next(val_iterator)
