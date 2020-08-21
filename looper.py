import numpy as np
import torch
from tqdm import tqdm


def dict_to_cuda(data, device):
    for key in data:
        if isinstance(data[key], dict):
            for k in data[key]:
                data[key][k] = data[key][k].to(device)
        else:
            data[key] = data[key].to(device)
    return data


class Logger:
    def __init__(self, metrics):
        self.metrics = {name: [] for name in metrics}

    def add(self, name):
        self.metrics[name] = []

    def update(self, name, value):
        self.metrics[name].append(value)

    def get(self, name):
        return self.metrics[name]

    def print_last(self):
        for name in self.metrics:
            print(f"{name} : {self.metrics[name][-1]}")

    def reset_all(self):
        for m in self.metrics:
            self.metrics[m] = []

    def reset(self, m):
        self.metrics[m] = []

    def avg(self, name, size):
        if name == 'loss':
            return sum(self.metrics[name][-size:]) / max(len(self.metrics[name]),1)
        return self.metrics[name][-1]


class Looper:
    def __init__(self, device, task, external_metrics=None):
        self.task = "cls"
        self.device = device
        self.logger = Logger(["loss"])
        self.external_metrics=external_metrics

        
    def train_step(self, model, optimizer, loss_fn, train_set):

        model.train()
        for batch in train_set:
            optimizer.zero_grad()
            batch = dict_to_cuda(batch, self.device)
            output = model(batch)
            loss = loss_fn(output, batch["target"])
            loss.backward()
            optimizer.step()
            self.logger.update("loss", loss.item())

    def val_step(self, model, val_funcs, val_set, epoch):

        model.eval()
        predictions = []
        true_labels = []
        self.num_b = 0
        with torch.no_grad():
            for batch in val_set:
                batch = dict_to_cuda(batch, self.device)
                self.num_b += 1
                output = model(batch)
                if self.task:
                    predictions += list(
                        torch.nn.functional.softmax(output, 1)
                        .detach()
                        .cpu()
                        .numpy()[:, 1]
                    )
                else:
                    raise NotImplementedError

                true_labels += list(batch["target"].cpu().numpy())
            for val_fn in val_funcs:
                name = val_fn.__name__
                if "acc" in name:
                    predictions = np.where(np.array(predictions) > 0.5, 1, 0)
                value = val_fn(true_labels, predictions)
                #print(name, value)
                if not self.external_metrics is None:
                    self.external_metrics(key=name, value=value, order=epoch)
                if name in self.logger.metrics:
                    self.logger.update(name, value)
                else:
                    self.logger.add(name)
                    self.logger.update(name, value)

    def train(
        self, model, epochs, train_set, val_set, optimizer, loss_fn, val_funcs,
    ):

        model.to(self.device)
        pbar = tqdm(total=epochs)
        for i in range(epochs):
            self.val_step(model, val_funcs, val_set, i)
            self.train_step(model, optimizer, loss_fn, train_set)
            metrics = {m: self.logger.avg(m, self.num_b) for m in self.logger.metrics}
            pbar.set_postfix(metrics)
            pbar.update(1)
#             self.logger.reset('accuracy_score')
#             self.logger.reset('roc_auc_score')

    def history(self):
        return self.logger.metrics
