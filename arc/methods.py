import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats.mstats import mquantiles
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import sys, math
from tqdm import tqdm

import sys
sys.path.insert(0, '..')
import arc

from arc.classification import ProbabilityAccumulator as ProbAccum

class SplitConformalEntropy:
    def __init__(self, X, Y, black_box, alpha, random_state=2020, allow_empty=True, verbose=False, box_name=None):
        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state, stratify=Y)
        self.black_box = black_box
        self.alpha = alpha
        self.allow_empty = allow_empty
        self.theta = np.random.randn(2)

        # Fit model
        self.black_box.fit(X_train, Y_train)
        
        # Grid search the optimal temperature by cross validation
        self.set_temperature(X_calib, Y_calib, alpha=alpha)
        self.calibrate(X_calib, Y_calib, alpha=alpha)
        
    def calibrate(self, X_calib, Y_calib, alpha, return_score=False):
        # Estimate probabilities on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib)
        
        # Adjust the probability prediction by temperature
        p_hat_calib = np.exp(np.log(p_hat_calib)/self.theta[0] + self.theta[1])
        p_hat_calib = p_hat_calib / p_hat_calib.sum(axis=1)[:,None]
        
        # Reweight by entropy
        p_hat_calib = p_hat_calib / np.array([Entropy(p) for p in p_hat_calib])[:, None]
        p_y_calib = np.array([p_hat_calib[i, Y_calib[i]] for i in range(len(Y_calib)) ])        
        
        # Compute threshold
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(X_calib.shape[0]))
        self.threshold_calibrated = mquantiles(p_y_calib, prob=1.0-level_adjusted)
        if return_score:
            return p_y_calib

    def ML(self, theta, X, Y):
        self.theta = theta
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        metric_optimal = 0
        metric = 0
        for train_index, test_index in cv.split(X, Y):
            B1 = self.calibrate(X[train_index], Y[train_index], alpha=self.alpha, return_score=True)
            B2 = self.calibrate(X[test_index], Y[test_index], alpha=self.alpha, return_score=True)
#             S_hat, B2 = self.predict(X[test_index], return_score=True)
            B1.sort(); B2.sort()
            M = B1[::5-1, None] - B2[None, :]
            metric += np.linalg.norm(M - np.diag(np.diag(M)))
        return metric
    
    def set_temperature(self, X, Y, alpha, n_folds=5):
#         X_calib, X_test, Y_calib, Y_test = train_test_split(X, Y, test_size=0.5, random_state=random_state, stratify=Y)

        optimal = minimize(self.ML, self.theta, args=(X, Y), options={'maxiter': 1000}, tol = 1e-3)
        self.theta = optimal.x
        print('status, nit, theta: ', optimal.success, optimal.nit, optimal.x)
            
#         if metric > metric_optimal:
#             print(f"conditional coverage={metric/n_folds} achieved at T={T}, C={C}")
#             metric_optimal = metric
        
    def predict(self, X, random_state=2020, return_score=False):
        p_hat = self.black_box.predict_proba(X)
        p_hat = np.exp(np.log(p_hat)/self.theta[0] + self.theta[1])
#         rng = np.random.default_rng(random_state)
#         p_hat += 1e-9 * rng.uniform(low=-1.0, high=1.0, size=p_hat.shape)
        p_hat = p_hat / p_hat.sum(axis=1)[:,None]
        p_hat = p_hat / np.array([Entropy(p) for p in p_hat])[:, None]       
        # Make prediction sets
        S_hat = [None]*X.shape[0]
        for i in range(X.shape[0]):
            S_hat[i] = np.where(p_hat[i,:] >= self.threshold_calibrated)[0]
            if (not self.allow_empty) and (len(S_hat[i])==0):
                S_hat[i] = [np.argmax(p_hat[i,:])]
        if return_score:
            return S_hat, p_hat
        return S_hat
    
class SplitConformalHomogeneous:
    def __init__(self, X, Y, black_box, alpha, random_state=2020, allow_empty=True, verbose=False, box_name=None):
        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state, stratify=Y)
        n2 = X_calib.shape[0]
        self.black_box = black_box
        self.alpha = alpha
        self.allow_empty = allow_empty

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # Estimate probabilities on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib)

        # Break ties at random
        rng = np.random.default_rng(random_state)
        p_hat_calib += 1e-9 * rng.uniform(low=-1.0, high=1.0, size=p_hat_calib.shape)
        p_hat_calib = p_hat_calib / p_hat_calib.sum(axis=1)[:,None]        
        p_y_calib = np.array([ p_hat_calib[i, Y_calib[i]] for i in range(len(Y_calib)) ])        
        
        # Compute threshold
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
        self.threshold_calibrated = mquantiles(p_y_calib, prob=1.0-level_adjusted)

    def predict(self, X, random_state=2020):
        n = X.shape[0]
        p_hat = self.black_box.predict_proba(X)
        # Break ties at random
        rng = np.random.default_rng(random_state)
        p_hat += 1e-9 * rng.uniform(low=-1.0, high=1.0, size=p_hat.shape)
        p_hat = p_hat / p_hat.sum(axis=1)[:,None]        
        # Make prediction sets
        S_hat = [None]*n
        for i in range(n):
            S_hat[i] = np.where(p_hat[i,:] >= self.threshold_calibrated)[0]
            if (not self.allow_empty) and (len(S_hat[i])==0):
                S_hat[i] = [np.argmax(p_hat[i,:])]
        return S_hat

def F1score(X, y, sets):
    TP, FP, TN, FN = 0, 0, 0, 0
    mclasses = len(set(y))
    for i in range(len(sets)):
        interval = sets[i]
        positives = [j for j in range(mclasses) if j in interval]
        negatives = [j for j in range(mclasses) if j not in interval]
        TP = TP + sum([1 for j in positives if j == y[i]])
        FP = FP + sum([1 for j in positives if j != y[i]])
        TN = TN + sum([1 for j in negatives if j != y[i]])  
        FN = FN + sum([1 for j in negatives if j == y[i]]) 
    F1 = 2 * TP/(2 * TP + FP + FN)
    return F1

class SplitConformal:
    def __init__(self, X, Y, black_box, alpha, random_state=2020, allow_empty=True, verbose=False, box_name=None):
        self.allow_empty = allow_empty

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state, stratify=Y)
        n2 = X_calib.shape[0]

        self.black_box = black_box

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib)
        grey_box = ProbAccum(p_hat_calib)

        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
        scores = alpha - alpha_max
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Store calibrate level
        self.alpha_calibrated = alpha - alpha_correction

    def predict(self, X, random_state=2020):
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon, allow_empty=self.allow_empty)
        return S_hat

def Entropy(p, eps=.0001):
    return np.where(p<=0, 0, -p*np.log(p)).sum()
#     return -sum([s * math.log(s + eps) for s in p])

def sigma(t):
    return 1/(np.exp(-t) + 1)

def quantile(v, alpha):
    m = int(np.ceil((1 - alpha) * (len(v) + 1))) - 1
    v = np.sort(v, axis = 0)
    return v[m]

def rescale(V, beta):
    S = np.exp(beta * V)
    return  S/np.expand_dims(np.sum(S, axis=1), axis=1)

class SplitConformalTransform:
    def __init__(self, X, Y, black_box, alpha, rho=0.01, model='ER', random_state=2020, allow_empty=True, verbose=False, box_name=None):
        self.allow_empty = allow_empty

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state, stratify=Y)
    
        self.K = len(set(Y))
        self.n = len(Y_train)
        self.alpha = alpha
        self.rho = rho
        self.npars = 2
        self.black_box = black_box
        self.X_calib = X_calib; self.Y_calib = Y_calib
        self.X_train = X_train; self.Y_train = Y_train

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib)
        n_calib = p_hat_calib.shape[0]
        
        # self.rg = self.approximator(p_hat_calib, X_calib, Y_calib, box_name=box_name)
        
        if model == 'ER':
            self.b = self.bER
        if model == 'HR':
            self.b = self.bHR
        if model == 'temp':
            self.b = self.bTemp
        if model == 'off':
            self.b = self.bOff
        if model == 'unweighted':
            self.b = self.bBase
            self.obj = None
            self.theta = None
            B = 1 - p_hat_calib
            level_adjusted = (1.0-self.alpha)*(1.0+1.0/float(B.shape[0]))
            self.threshold_calibrated = mquantiles(np.array([-B[i, Y_calib[i]] for i in range(len(Y_calib))]), prob=1-level_adjusted)
        if model != 'unweighted':
            self.fit_calibrate(p_hat_calib, Y_calib)

    def fit_calibrate(self, prob, y):
        prob_train, prob_calib, Y_train, Y_calib = train_test_split(prob, y, test_size=0.5, stratify=y)
        g = ProbAccum(prob_train).calibrate_scores_approximator()
        self.theta = np.linalg.solve(g.T @ g, g.T @ np.ones((prob_train.shape[0], 1)))
        level_adjusted = (1.0 - self.alpha) * (1.0 + 1.0 / float(prob_calib.shape[0]))
        A = 1 - prob_calib
        g = ProbAccum(prob_calib).calibrate_scores_approximator()
        B = A * np.exp(-(1 - g @ self.theta) ** 2)
        self.threshold_calibrated = mquantiles(np.array([-B[i, Y_calib[i]] for i in range(len(Y_calib))]),
                                               prob=1 - level_adjusted)

    def predict(self, X_test):
        prob = self.black_box.predict_proba(X_test)
        A = 1 - prob
        if self.theta is None:
            B = A
        else:
            g = ProbAccum(prob).calibrate_scores_approximator()
            B = A * np.exp(-(1 - g @ self.theta) ** 2)
        S_hat = [None]*X_test.shape[0]
        for i in range(X_test.shape[0]):
            S_hat[i] = np.where(-B[i,:] >= self.threshold_calibrated)[0]
            if (not self.allow_empty) and (len(S_hat[i])==0):
                S_hat[i] = [np.argmin(B[i,:])]
        # a = 1
        return S_hat

        # X, y = self.X_calib, self.Y_calib
        # X, A, G, h = self.prepareData(self.black_box.predict_proba(X), X)
        # B = self.b(A, G, h, self.theta)
        # q = quantile([B[i, y[i]] for i in range(len(y))], self.alpha)
        #
        # X = X_test
        # X, A, G, h = self.prepareData(self.black_box.predict_proba(X), X)
        # B = self.b(A, G, h, self.theta)
        # sets = [[m for m in range(self.K) if B[i][m] < q] for i in range(X.shape[0])]
        # return sets
        
    def basicScore(self, prob):
        return 1 - prob

    def fitTransformation(self, prob, X, y, prob2, X2, y2):
        d1 = self.prepareData(prob, X, y)
        d2 = self.prepareData(prob2, X2, y2)
        initial_guess =  .5 * np.random.randn(self.npars)
#         optimal = differential_evolution(self.obj, [(-5, 5), (-5, 5)], args=(d1, d2))
        optimal = minimize(self.obj, initial_guess,
                args=(d1, d2), options={'maxiter': 1000}, method='L-BFGS-B', tol = 1e-3)
        print('status, nit, theta: ', optimal.success, optimal.nit, optimal.x)
        return optimal.x
 
    def ML(self, theta, d1, d2):
        [X1, A1, G1, h1, y1] = d1
        [X2, A2, G2, h2, y2] = d2
        B1 = self.b(A1, G1, h1, theta)
        b1 = np.array([B1[i, y1[i]] for i in range(len(y1))])
        b1 = np.expand_dims(b1, axis = 1)
        B2 = self.b(A2, G2, h2, theta)
        b2 = np.array([B2[i, y2[i]] for i in range(len(y2))])
        b2 = np.expand_dims(b2, axis = 1)
        b1.sort(); b2.sort()
        M = b1 - b2.transpose()
        m = np.linalg.norm(M - np.diag(np.diag(M)))
        ell = m/len(y2)
        return ell + self.rho * theta @ theta
    
    def prepareData(self, prob, X, y):
        A = self.basicScore(prob)
        g = ProbAccum(prob)
        G = g.calibrate_scores_approximator(y)[:, None]
        # G = np.array([self.rg[i].predict(X) for i in range(self.K)]).transpose()
        h = np.array([Entropy(p) for p in prob])[:, None]
#         h = np.array([H(1 - G[i, :]).squeeze() for i in range(X.shape[0])])
        return X, A, G, h

    def bER(self, A, G, h, theta):
        r = theta[0]**2 + theta[1]**2 * G
        return A * np.exp(-r)
    
    def bBase(self, A, G, h, theta):
        return A
    
    def bHR(self, A, G, p, theta):
        p = np.exp((np.log(p) + theta[0]) / theta[1]**2)
        p = p / p.sum(axis=1)[:, np.newaxis]
        h = np.array([Entropy(pi) for pi in p])
        H = np.expand_dims(h, axis=1)
#         print(A, H)
        # r = theta[0]**2 + theta[1]**2 * H
        return A/(1e-4 + H)
    
    def bTemp(self, A, G, h, theta):
        r1 = theta[0]**2 * A#1 - rescale(1 - A, theta[0]**2)
        r2 = 1 - rescale(1 - G, theta[1]**2)
        return r1 + r2
    
    def bOff(self, A, G, h, theta):
        r1 = A + theta[1] * G
        r2 = sum(theta**2)
        return r1/r2

    def approximator(self, prob, X, y, box_name=None):
        g = ProbAccum(prob)
        return g.calibrate_scores_approximator(y)
        # A = self.basicScore(prob)
        # a = [A[i, y[i]] for i in range(self.n)]
        # rf = []
        # for m in range(self.K):
        #     rf.append(RandomForestRegressor(max_depth=25, random_state=0))
        #     xm = np.array([X[i] for i in range(self.n) if y[i] == m])
        #     am = np.array([a[i] for i in range(self.n) if y[i] == m])
        #     rf[-1].fit(xm, am.squeeze())
        # return rf

    
class SplitConformalTransformOld:
    def __init__(self, X, Y, black_box, alpha, obj='ML', rho=0.1, random_state=2020, allow_empty=True, verbose=False):
        self.allow_empty = allow_empty

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state, stratify=Y)
        n2 = X_calib.shape[0]
    
        self.K = len(set(Y))
        self.n = len(Y_train)
        self.alpha = alpha
        self.rho = rho
        self.black_box = black_box
        self.X_calib = X_calib; self.Y_calib = Y_calib
        self.X_train = X_train; self.Y_train = Y_train

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib)
        self.rg = self.approximator(p_hat_calib, X_train, Y_train)
        
        if obj == 'ML': 
            self.obj = self.ML
            self.theta = self.fitTransformation(p_hat_calib, X_train, Y_train)
        if obj == 'size': 
            self.obj = self.smoothSize
            self.theta = self.fitTransformation(p_hat_calib, X_train, Y_train)
        if obj == 'unweighted': 
            self.obj = None
            self.theta = numpy.zeros(10)
    
    def predict(self, X_test):
        X, y = self.X_calib, self.Y_calib
        X, A, G, h = self.prepareData(self.black_box.predict_proba(X), X)
        B = self.b(A, G, h, self.theta)
        q = quantile([B[i, y[i]] for i in range(len(y))], self.alpha)
        
        X = X_test
        X, A, G, h = self.prepareData(self.black_box.predict_proba(X), X)
        B = self.b(A, G, h, self.theta)
        sets = [[m for m in range(self.K) if B[i][m] < q] for i in range(X.shape[0])]
        return sets
#         F1 = F1score(sets, y)
#         sizes = numpy.sum([len(sets[i]) for i in range(len(sets))])/len(y)
#         val = numpy.sum([1 for i in range(len(y)) if y[i] in sets[i]])/len(y)
#         return sizes, F1, val
        
    def basicScore(self, prob):
        return 1 - prob

    def approximator(self, prob, X, y):
        A = self.basicScore(prob)
        a = [A[i, y[i]] for i in range(self.n)]
        rf = []
        for m in range(self.K): 
            rf.append(RandomForestRegressor(max_depth=25, random_state=0))
            xm = np.array([X[i] for i in range(self.n) if y[i] == m])
            am = np.array([a[i] for i in range(self.n) if y[i] == m])
            rf[-1].fit(xm, am.squeeze())
        return rf    
    
    def fitTransformation(self, prob, X, y):
        d1 = self.prepareData(prob, X), y
        initial_guess =  .1 * np.random.randn(10)
        optimal = minimize(self.obj, initial_guess,
                args=(d1, d1), options={'maxiter': 1000}, tol = 1e-3)
        print('status, nit, theta: ', optimal.success, optimal.nit, optimal.x)
        return optimal.x
 
    def ML(self, theta, d1, d2):
        [X1, A1, G1, h1], y1 = d1
        [X2, A2, G2, h2], y2 = d2
        B1 = self.b(A1, G1, h1, theta)
        b1 = np.array([B1[i, y1[i]] for i in range(self.n)])
        b1 = np.expand_dims(b1, axis = 1)
        B2 = self.b(A2, G2, h2, theta)
        b2 = np.array([B2[i, y2[i]] for i in range(self.n)])
        b2 = np.expand_dims(b2, axis = 1)
        M = b1 - b2.transpose()
        m = np.linalg.norm(M - np.diag(np.diag(M))) 
        ell = m/self.n
        return ell + self.rho * theta @ theta
    
    def smoothSize(self, theta, d1, d2, scale = 2):
        [X1, A1, G1, h1], y1 = d1
        [X2, A2, G2, h2], y2 = d2
        beta = 2
        B1 = self.b(A1, G1, h1, theta)
        B2 = self.b(A2, G2, h2, theta)
        b2 = np.array([B2[i2, y2[i2]] for i2 in range(self.n)])
        w = b2/sum(b2)
        w = np.exp(beta * w) + 1e-4
        w = np.diag(w / sum(w))
        B1 = np.expand_dims(B1, axis = 2)
        B1 = np.transpose(B1, axes = [0, 2, 1])
        b2 = np.array(b2)
        b2 = np.expand_dims(b2, axis = [0, 2])
        S = np.sum(sigma(scale * (-(B1 - b2))), axis = 2)
        ell = 1/self.n * numpy.sum(S @ w)
        return ell + self.rho * theta @ theta
    
    def prepareData(self, prob, X):
        A = self.basicScore(prob)
        G = np.array([self.rg[i].predict(X) for i in range(self.K)]).transpose()
        h = np.array([H(1 - G[i, :]).squeeze() for i in range(X.shape[0])])
        return X, A, G, h

    def r(self, t, G, h):
        h = np.expand_dims(h, axis=1)
        r1, r2 = [t[0 + i] + 
                t[2 + i] * np.power(abs(G), t[4 + i]) + 
                t[6 + i] * np.power(abs(h), t[8+ i]) for i in [0, 1]]
        return r1, r2
    
    def b(self, A, G, h, theta):
        r1, r2 = self.r(theta, G, h)
        return A * np.exp(-r1) - r2 
    
    