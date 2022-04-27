from torchmetrics import MeanAbsoluteError, MeanSquaredError


class Metrics():
    def __init__(self, device, metrics=None):
        if metrics is None:
            metrics = {'MAE': MeanAbsoluteError(),
                       'RMSE': MeanSquaredError(squared=False),
                       'MSE': MeanSquaredError(squared=True)}
        self.metrics = metrics
        for name, metric in self.metrics.items():
            metric.to(device)

    def step(self, prediction, target):
        for name, metric in self.metrics.items():
            metric(prediction, target)

    def compute(self, run):
        for name, metric in self.metrics.items():
            value = metric.compute().item()
            run.log(name, value)
            print(name, value)

    def reset(self):
        for name, metric in self.metrics.items():
            metric.reset()
