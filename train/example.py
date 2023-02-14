import ai
import ai.model as m

model = m.Model(m.seq(m.flatten(), m.fc(784, 10))).init()

trainer = ai.Trainer(
    ai.train.Classify(),
    ai.data.mnist_dataset().loader(batch_size=32, device='cpu'),
)

trainer.train(model, ai.opt.sgd(model))
