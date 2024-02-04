
import os
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from scipy.stats.mstats import mquantiles
from run_experiments_new import assess_predictions


# def seed_everything(seed=2021):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#
# seed_everything(2024)

def entropy(logits):
    full_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    full_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    ent = -torch.sum(full_probs * full_logprobs, dim=-1)
    return ent

def compute_coverage_loss(confidence_sets, labels, alpha, transform=torch.square):
    one_hot_labels = nn.functional.one_hot(labels, confidence_sets.shape[1])
    return torch.mean(transform(torch.sum(confidence_sets * one_hot_labels, dim=1) - (1 - alpha)))

def compute_hinge_size_loss(confidence_sets, target_size=1, transform=torch.log):
    return torch.mean(transform
        (torch.maximum(torch.sum(confidence_sets, dim=1) - target_size, torch.zeros(confidence_sets.shape[0]))))

class MnistClassifierConformal(nn.Module):
    def __init__(self, alpha=0.1, cover_transform_fn=torch.square, size_transform_fn=torch.log1p):
        super(MnistClassifierConformal, self).__init__()
        self.alpha = alpha
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.temperature = nn.Parameter(torch.ones(1 ) *1.5)
        self.cover_transform_fn = cover_transform_fn
        self.size_transform_fn = size_transform_fn
        self.cal_data = None
        params = []
        for module in [self.conv1, self.conv2, self.fc1, self.fc2]:
            params += module.parameters()
        self.optimizer_network = optim.Adam(params + [self.temperature], lr=0.001)
        self.optimizer_temperature = optim.Adam([self.temperature], lr=0.001)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x)
        return self.temperature_scale(x)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def train_conformal(self, cal_loader, fraction=0.5):
        for module in [self.conv1, self.conv2, self.fc1, self.fc2]:
            for param in module.parameters():
                param.requires_grad = False
        cal_x = [];
        cal_y = []
        for idx, (x, label) in enumerate(cal_loader):
            self.optimizer_temperature.zero_grad()
            x_calib, y_calib = x[:int(x.shape[0] * fraction)], label[:int(x.shape[0] * fraction)]
            cal_x.append(x_calib);
            cal_y.append(y_calib)
            logits = self.forward(x_calib)
            p_y_calib = torch.Tensor([logits[i, y_calib[i]] for i in range(len(y_calib))])
            # p_y_calib = nn.parameter.Parameter(data=torch.Tensor([logits[i, y_calib[i]] for i in range(len(y_calib))]),
            #                                    requires_grad=True)
            level_adjusted = (1.0 - self.alpha) * (1.0 + 1.0 / float(len(y_calib)))
            self.threshold_calibrated = torch.FloatTensor(mquantiles(p_y_calib, prob=1.0 - level_adjusted))

            x_calib_test, y_calib_test = x[int(x.shape[0] * fraction):], label[int(x.shape[0] * fraction):]
            conformal_set = self.predict(x_calib_test)
            coverage_loss = compute_coverage_loss(conformal_set, y_calib_test, alpha=self.alpha,
                                                  transform=self.cover_transform_fn)
            efficiency_loss = compute_hinge_size_loss(conformal_set, target_size=1, transform=self.size_transform_fn)
            if idx % 10 == 1:
                print(
                    f"coverage_loss={coverage_loss}, efficiency_loss={efficiency_loss}, temperature={float(self.temperature)} at {idx}-th iteration")
            loss = coverage_loss + efficiency_loss
            # loss = efficiency_loss
            loss.backward()
            self.optimizer_temperature.step()

        if self.cal_data == None:
            self.cal_data = torch.utils.data.DataLoader(dataset=[cal_x, cal_y], shuffle=True, batch_size=len(cal_x))

    def predict(self, X, random_state=2020):
        n = X.shape[0]
        logit = self.forward(X)
        S_hat = nn.functional.sigmoid(logit - self.threshold_calibrated)
        # S_hat = torch.zeros((n, logit.shape[-1]))
        # for i in range(n):
        # S_hat[i][torch.where(logit[i ,:] >= self.threshold_calibrated)[0]] = 1
        return S_hat

    def predict_all_cal(self, X, cal_loader=None):
        if cal_loader != None:
            cal_x = []
            cal_y = []
            for x_calib, y_calib in cal_loader:
                cal_x.append(x_calib)
                cal_y.append(y_calib)
            self.cal_data = torch.utils.data.DataLoader(dataset=[cal_x, cal_y], shuffle=True, batch_size=len(cal_x))
        x_calib = torch.cat(self.cal_data.dataset[0], dim=0)
        y_calib = torch.cat(self.cal_data.dataset[1], dim=0)
        logits = self.forward(x_calib)
        p_y_calib = torch.Tensor([logits[i, y_calib[i]] for i in range(len(y_calib))])
        level_adjusted = (1.0 - self.alpha) * (1.0 + 1.0 / float(len(y_calib)))
        self.threshold_calibrated = torch.FloatTensor(mquantiles(p_y_calib, prob=1.0 - level_adjusted))
        logit = self.forward(X)
        if cal_loader != None:
            S_hat = torch.zeros_like(logit)
            for i in range(S_hat.shape[0]):
                S_hat[i][torch.where(logit[i, :] >= self.threshold_calibrated)[0]] = 1
        else:
            S_hat = nn.functional.sigmoid(logit - self.threshold_calibrated)
        return S_hat

    def train_base(self, train_loader):
        nll_criterion = nn.CrossEntropyLoss()
        for idx, (x, label) in enumerate(train_loader):
            self.optimizer_network.zero_grad()
            logit = self.forward(x)
            loss = nll_criterion(logit, label)
            if idx % 50 == 1:
                print(f"loss={float(loss)} , temperature={float(self.temperature)} at {idx}-th training iteration")
            loss.backward()
            self.optimizer_network.step()

def train_model(CP, alpha=0.1, cover_transform_fn=torch.square, size_transform_fn=torch.log1p, random_state=2024):
    train_subset, val_subset, cal_subset, test_subset = torch.utils.data.random_split(
        datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
        [20000, 18000, 20000, 2000], generator=torch.Generator().manual_seed(random_state))

    train_loader = torch.utils.data.DataLoader(dataset=train_subset, shuffle=True, batch_size=50)
    cal_loader = torch.utils.data.DataLoader(dataset=cal_subset, shuffle=False, batch_size=1000)
    test_loader = torch.utils.data.DataLoader(dataset=test_subset, shuffle=False, batch_size=len(test_subset.indices))

    classifier = MnistClassifierConformal(alpha=alpha, cover_transform_fn=cover_transform_fn, size_transform_fn=size_transform_fn)
    for epoch in range(1):
        print(f"Training epoch {epoch} ...")
        classifier.train_base(train_loader)
    if CP:
        for epoch in range(1):
            classifier.train_conformal(cal_loader, fraction=0.8)

    classifier.eval()
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        if CP:
            conformal_set = classifier.predict_all_cal(data).detach().numpy()
            S_hat = [np.where(conformal_set[i] > 1 / 2)[0] for i in range(conformal_set.shape[0])]
        else:
            conformal_set = classifier.predict_all_cal(data, cal_loader).detach().numpy()
            S_hat = [np.nonzero(conformal_set[i])[0] for i in range(conformal_set.shape[0])]

    result = assess_predictions(S_hat, data.detach().numpy().reshape(data.shape[0], -1), target.detach().numpy())
    return result


result1 = train_model(True)
result2 = train_model(False)
# print(result1, result2)

result1

# train_subset, val_subset, cal_subset, test_subset = torch.utils.data.random_split(
#     datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()), [15000, 15000, 15000, 15000], generator=torch.Generator().manual_seed(1))
#
# train_loader = torch.utils.data.DataLoader(dataset=train_subset, shuffle=True, batch_size=50)
# cal_loader = torch.utils.data.DataLoader(dataset=cal_subset, shuffle=False, batch_size=500)
# test_loader = torch.utils.data.DataLoader(dataset=test_subset, shuffle=False, batch_size=len(test_subset.indices))
#
#
# classifier = MnistClassifierConformal(alpha=0.1)
# cover_transform_fn_list = [torch.abs, torch.square]
# size_transform_fn_list = [torch.log1p, torch.abs, torch.square]
#
# for epoch in range(2):
#     print(f"Training epoch {epoch} ...")
#     classifier.train_base(train_loader)
#
# for epoch in range(3):
#     classifier.train_conformal(cal_loader, fraction=0.8)
#
# classifier.eval()
# for data, target in test_loader:
#     data, target = Variable(data), Variable(target)
#     conformal_set = classifier.predict(data).detach().numpy()
#     S_hat = [np.where(conformal_set[i]>1/2)[0] for i in range(conformal_set.shape[0])]
#     # S_hat = [np.nonzero(conformal_set[i])[0] for i in range(conformal_set.shape[0])]
#     result = assess_predictions(S_hat, data.detach().numpy().reshape(data.shape[0], -1), target.detach().numpy())
