import torch


class Task:
    def __call__(s, model):
        model.eval()
        with torch.no_grad():
            evaluation = s.evaluate(model)
        return evaluation

    def evaluate(s, model):
        raise NotImplementedError()


class Classify(Task):
    def __init__(s, data):
        s._data = data

    def evaluate(s, model):
        correct = 0
        total = 0
        for batch in s._data:
            x, y = batch['x'], batch['y']
            output = model(x)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
            total += x.shape[0]
        assert total > 0, 'empty data in classify task'
        return 100. * correct / total
