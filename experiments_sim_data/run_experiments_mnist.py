
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
from os import path


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

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

def uncertainty(logits, method='entropy'):
    full_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    full_probs = torch.where(full_probs<=0, torch.finfo().eps, full_probs)
    if method == 'entropy':
        full_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        ent = -torch.sum(full_probs * full_logprobs, dim=-1)
        return ent
    elif method == 'gini':
        gini = (full_probs * (1 - full_probs)).sum(dim=1)
        return gini
        # n = full_probs.shape[1]
        # full_probs_sorted = torch.sort(full_probs, dim=1)[0]  # values must be sorted
        # index = torch.arange(1, n + 1)
        # gini = torch.sum((2 * index - n - 1).unsqueeze(0) * full_probs_sorted, dim=1) / (n * torch.sum(full_probs_sorted, dim=1))
        # return 1 - gini

def compute_coverage_loss(confidence_sets, labels, alpha, transform=torch.square):
    one_hot_labels = nn.functional.one_hot(labels, confidence_sets.shape[1])
    return torch.mean(transform(torch.sum(confidence_sets * one_hot_labels, dim=1) - (1 - alpha)))

def compute_hinge_size_loss(confidence_sets, target_size=1, transform=torch.log):
    return torch.mean(transform
        (torch.maximum(torch.sum(confidence_sets, dim=1) - target_size, torch.zeros(confidence_sets.shape[0]))))

class MnistClassifier(nn.Module):
    def __init__(self, alpha=0.1):
        super(MnistClassifier, self).__init__()
        self.alpha = alpha
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)
        params = []
        for module in [self.conv1, self.conv2, self.fc1, self.fc2]:
            params += module.parameters()
        self.optimizer_network = optim.Adam(params, lr=0.0001)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    def train_base(self, train_loader):
        nll_criterion = nn.CrossEntropyLoss()
        for idx, (x, label) in enumerate(train_loader):
            self.optimizer_network.zero_grad()
            logit = self.forward(x)
            loss = nll_criterion(logit, label)
            # if idx % 50 == 1:
            #     print(f"loss={float(loss)} at {idx}-th training iteration")
            loss.backward()
            self.optimizer_network.step()
    def predict_all_cal(self, X, cal_loader):
        cal_x = []
        cal_y = []
        for x_calib, y_calib in cal_loader:
            cal_x.append(x_calib)
            cal_y.append(y_calib)
        self.cal_data = torch.utils.data.DataLoader(dataset=[cal_x, cal_y], shuffle=True, batch_size=len(cal_x))
        x_calib = torch.cat(self.cal_data.dataset[0], dim=0)
        y_calib = torch.cat(self.cal_data.dataset[1], dim=0)
        logits = self.forward(x_calib)
        probs = torch.softmax(logits, dim=1)
        p_y_calib = torch.Tensor([probs[i, y_calib[i]] for i in range(len(y_calib))])
        level_adjusted = (1.0 - self.alpha) * (1.0 + 1.0 / float(len(y_calib)))
        self.threshold_calibrated = torch.FloatTensor(mquantiles(p_y_calib, prob=1.0 - level_adjusted))
        logit = self.forward(X)
        prob = torch.softmax(logit, dim=1)
        S_hat = torch.zeros_like(logit)
        for i in range(S_hat.shape[0]):
            S_hat[i][torch.where(prob[i, :] >= self.threshold_calibrated)[0]] = 1
        return S_hat
class MnistClassifierConformal(nn.Module):
    def __init__(self, mnist_classifier, alpha=0.1, uncertainty_method='entropy', cover_transform_fn=torch.square, size_transform_fn=torch.log1p):
        super(MnistClassifierConformal, self).__init__()
        self.mnist_classifier = mnist_classifier
        self.mnist_classifier.eval()
        for param in self.mnist_classifier.parameters():
            param.requires_grad = False
        self.alpha = alpha
        self.uncertainty_method = uncertainty_method
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.temperature = nn.Parameter(torch.ones(1)*1.5)
        self.cover_transform_fn = cover_transform_fn
        self.size_transform_fn = size_transform_fn
        self.cal_data = None
        # self.optimizer_temperature = optim.LBFGS([self.temperature], lr=0.01, max_iter=200, tolerance_grad=1e-8) #optim.Adam([self.temperature], lr=0.01) #

    def forward(self, x):
        with torch.no_grad():
            x = self.mnist_classifier(x)
        return self.temperature_scale(x)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / torch.abs(temperature)

    def train_conformal(self, cal_loader, fraction=0.5):
        for module in [self.conv1, self.conv2, self.fc1, self.fc2]:
            for param in module.parameters():
                param.requires_grad = False
        cal_x = []
        cal_y = []
        for idx, (x, label) in enumerate(cal_loader):
            self.optimizer_temperature.zero_grad()
            x_calib, y_calib = x[:int(x.shape[0] * fraction)], label[:int(x.shape[0] * fraction)]
            cal_x.append(x_calib)
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
        logits = torch.softmax(logits, dim=1)
        # logits = logits / uncertainty(logits, method=self.uncertainty_method)[:, None] # entropy(nn.functional.softmax(logits, dim=1))[:, None]
        # p_y_calib = torch.Tensor([logits[i, y_calib[i]] for i in range(len(y_calib))])
        # probs = torch.softmax(logits, dim=1) / uncertainty(logits, method=self.uncertainty_method)[:, None] # entropy(nn.functional.softmax(logits, dim=1))[:, None]
        p_y_calib = torch.Tensor([logits[i, y_calib[i]] for i in range(len(y_calib))])
        level_adjusted = (1.0 - self.alpha) * (1.0 + 1.0 / float(len(y_calib)))
        self.threshold_calibrated = torch.FloatTensor(mquantiles(p_y_calib, prob=1.0 - level_adjusted))
        logit = self.forward(X)
        logit = torch.softmax(logit, dim=1)
        # logit = logit / uncertainty(logit, method=self.uncertainty_method)[:, None]
        # prob = torch.softmax(logit, dim=1) / uncertainty(logit, method=self.uncertainty_method)[:, None]
        if cal_loader != None:
            S_hat = torch.zeros_like(logit)
            for i in range(S_hat.shape[0]):
                S_hat[i][torch.where(logit[i, :] >= self.threshold_calibrated)[0]] = 1
        # else:
        #     S_hat = nn.functional.sigmoid(logit - self.threshold_calibrated)
        return S_hat

    def train_base(self, val_loader):
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()

        def eval():
            optimizer.zero_grad()
            logit = self.forward(x)
            loss = nll_criterion(logit, label)
            loss.backward()
            return loss

        # print(f"temperature={float(self.temperature)}")
        for idx, (x, label) in enumerate(val_loader):
            ece = ece_criterion(self.forward(x), label)
            print(f"ECE loss={float(ece)} before temperature scaling, T={abs(float(self.temperature))}")
            optimizer = optim.LBFGS([self.temperature], lr=0.02, max_iter=200)
            optimizer.step(eval)
        ece = ece_criterion(self.forward(x), label)
        print(f"ECE loss={float(ece)} after temperature scaling, T={abs(float(self.temperature))}")
        # print(f"temperature={float(self.temperature)}")
            # self.optimizer_temperature.zero_grad()
            # logit = self.forward(x)
            # loss = nll_criterion(logit, label)
            # if idx == 0:
            #     ece = ece_criterion(self.forward(x), label)
            #     print(f"ECE loss={float(ece)} before temperature scaling, T={float(self.temperature)}")
            # if idx % 50 == 1:
            #     print(f"loss={float(loss)} , temperature={float(self.temperature)} at {idx}-th training iteration")
            # loss.backward()
            # self.optimizer_temperature.step()
        # ece = ece_criterion(self.forward(x), label)
        # print(f"ECE loss={float(ece)} after temperature scaling, T={float(self.temperature)}")

def train_model(alpha=0.1, uncertainty_method='entropy', cover_transform_fn=torch.square, size_transform_fn=torch.log1p, random_state=2024):
    train_subset, val_subset, cal_subset, test_subset, left_subset = torch.utils.data.random_split(
        datasets.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor()),
        [20000, 1000, 30000, 1000, 8000], generator=torch.Generator().manual_seed(random_state))

    train_loader = torch.utils.data.DataLoader(dataset=train_subset, shuffle=False, batch_size=100)
    val_loader = torch.utils.data.DataLoader(dataset=val_subset, shuffle=False, batch_size=len(val_subset.indices))
    cal_loader = torch.utils.data.DataLoader(dataset=cal_subset, shuffle=False, batch_size=len(cal_subset.indices))
    test_loader = torch.utils.data.DataLoader(dataset=test_subset, shuffle=False, batch_size=len(test_subset.indices))

    classifier = MnistClassifier(alpha=alpha)
    for epoch in range(1):
        print(f"Training epoch {epoch} ...")
        classifier.train_base(train_loader)
    classifier.eval()
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
    conformal_set = classifier.predict_all_cal(data, cal_loader).detach().numpy()
    S_hat = [np.nonzero(conformal_set[i])[0] for i in range(conformal_set.shape[0])]
    result_base = assess_predictions(S_hat, data.detach().numpy().reshape(data.shape[0], -1), target.detach().numpy())

    classifier_conformal = MnistClassifierConformal(mnist_classifier=classifier, uncertainty_method=uncertainty_method, alpha=alpha)
    for epoch in range(1):
        print(f"Temperature training epoch {epoch} on a validation set...")
        classifier_conformal.train_base(val_loader)
    classifier_conformal.eval()
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
    conformal_set = classifier_conformal.predict_all_cal(data, cal_loader).detach().numpy()
    S_hat = [np.nonzero(conformal_set[i])[0] for i in range(conformal_set.shape[0])]
    result_temperature = assess_predictions(S_hat, data.detach().numpy().reshape(data.shape[0], -1), target.detach().numpy())

    return result_base, result_temperature


def run_experiment(out_dir):
    out_file = out_dir + "/summary.csv"
    if path.exists(out_file):
        results = pd.read_csv(out_file)
    else:
        results = pd.DataFrame()
    # List of calibration methods to be compared
    methods = ['baseline', 'reweighting']
    uncertainty_methods = ['entropy', 'gini']
    alphas = [0.05, 0.1, 0.2]
    experiments = np.arange(10)
    for alpha in alphas:
        for experiment in experiments:
            for uncertainty_method in uncertainty_methods:
                res_base, res_temperature = train_model(alpha=alpha, uncertainty_method=uncertainty_method, random_state=int(2024+experiment))
                res_base['Method'] = methods[0]
                res_temperature['Method'] = methods[1]
                res_base['Uncertainty'] = ''
                res_temperature['Uncertainty'] = uncertainty_method
                res_base['Experiment'] = experiment
                res_temperature['Experiment'] = experiment
                res_base['Nominal'] = 1 - alpha
                res_temperature['Nominal'] = 1 - alpha
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                results = pd.concat([results, res_base, res_temperature])
                results.to_csv(out_file, index=False, float_format="%.4f")
                print(f"Updated summary of results on alpha={alpha}, uncertainty={uncertainty_method}, cc={res_base['Conditional coverage'].item()}vs{res_temperature['Conditional coverage'].item()}")

out_dir = "/Users/lorry/Documents/20240125 conformal prediction/arc_new/arc/results"
run_experiment(out_dir)