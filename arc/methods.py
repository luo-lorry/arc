import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.stats.mstats import mquantiles
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
import sys
from tqdm import tqdm

from arc.classification import ProbabilityAccumulator as ProbAccum

class CVPlus:
    def __init__(self, X, Y, black_box, alpha, n_folds=10, random_state=2020, allow_empty=True, verbose=False):
        X = np.array(X)
        Y = np.array(Y)
        self.black_box = black_box
        self.n = X.shape[0]
        self.classes = np.unique(Y)
        self.n_classes = len(self.classes)
        self.n_folds = n_folds
        self.cv = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        self.alpha = alpha
        self.allow_empty = allow_empty
        self.verbose = verbose
        
        # Fit prediction rules on leave-one out datasets
        self.mu_LOO = [ black_box.fit(X[train_index], Y[train_index]) for train_index, _ in self.cv.split(X) ]

        # Accumulate probabilities for the original data with the grey boxes
        test_indices = [test_index for _, test_index in self.cv.split(X)]
        self.test_indices = test_indices
        self.folds = [[]]*self.n
        for k in range(self.n_folds):
            for i in test_indices[k]:
                self.folds[i] = k
        self.grey_boxes = [[]]*self.n_folds
        if self.verbose:
            print("Training black boxes on {} samples with {}-fold cross-validation:".
                  format(self.n, self.n_folds), file=sys.stderr)
            sys.stderr.flush()
            for k in tqdm(range(self.n_folds), ascii=True, disable=True):
                self.grey_boxes[k] = ProbAccum(self.mu_LOO[k].predict_proba(X[test_indices[k]]))
        else:
            for k in range(self.n_folds):
                self.grey_boxes[k] = ProbAccum(self.mu_LOO[k].predict_proba(X[test_indices[k]]))
               
        # Compute scores using real labels
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=self.n)
        self.alpha_max = np.zeros((self.n, 1))
        if self.verbose:
            print("Computing scores for {} samples:". format(self.n), file=sys.stderr)
            sys.stderr.flush()
            for k in tqdm(range(self.n_folds), ascii=True, disable=True):
                idx = test_indices[k]
                self.alpha_max[idx,0] = self.grey_boxes[k].calibrate_scores(Y[idx], epsilon=epsilon[idx])
        else:
            for k in range(self.n_folds):
                idx = test_indices[k]
                self.alpha_max[idx,0] = self.grey_boxes[k].calibrate_scores(Y[idx], epsilon=epsilon[idx])
            
    def predict(self, X, random_state=2020):
        n = X.shape[0]
        S = [[]]*n
        n_classes = len(self.classes)

        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)
        prop_smaller = np.zeros((n,n_classes))

        if self.verbose:
            print("Computing predictive sets for {} samples:". format(n), file=sys.stderr)
            sys.stderr.flush()
            for fold in tqdm(range(self.n_folds), ascii=True, disable=True):
                gb = ProbAccum(self.mu_LOO[fold].predict_proba(X))
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    for i in self.test_indices[fold]:
                        prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])
        else:
            for fold in range(self.n_folds):
                gb = ProbAccum(self.mu_LOO[fold].predict_proba(X))
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    for i in self.test_indices[fold]:
                        prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])

        for k in range(n_classes):
            prop_smaller[:,k] /= float(self.n)
                
        level_adjusted = (1.0-self.alpha)*(1.0+1.0/float(self.n))
        S = [None]*n
        for i in range(n):
            S[i] = np.where(prop_smaller[i,:] < level_adjusted)[0]
            if (not self.allow_empty) and (len(S[i])==0): # Note: avoid returning empty sets
                if len(S[i])==0:
                    S[i] = [np.argmin(prop_smaller[i,:])]            
        return S

class JackknifePlus:
    def __init__(self, X, Y, black_box, alpha, random_state=2020, allow_empty=True, verbose=False):
        self.black_box = black_box
        self.n = X.shape[0]
        self.classes = np.unique(Y)
        self.alpha = alpha
        self.allow_empty = allow_empty
        self.verbose = verbose

        # Fit prediction rules on leave-one out datasets
        self.mu_LOO = [[]] * self.n
        if self.verbose:
            print("Training black boxes on {} samples with the Jacknife+:". format(self.n), file=sys.stderr)
            sys.stderr.flush()
            for i in range(self.n):
                print("{} of {}...".format(i+1, self.n), file=sys.stderr)
                sys.stderr.flush()
                self.mu_LOO[i] = black_box.fit(np.delete(X,i,0),np.delete(Y,i))
        else:
            for i in range(self.n):
                self.mu_LOO[i] = black_box.fit(np.delete(X,i,0),np.delete(Y,i))

        # Accumulate probabilities for the original data with the grey boxes
        self.grey_boxes = [ ProbAccum(self.mu_LOO[i].predict_proba(X[i])) for i in range(self.n) ]

        # Compute scores using real labels
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=self.n)
        
        self.alpha_max = np.zeros((self.n, 1))    
        if self.verbose:
            print("Computing scores for {} samples:". format(self.n), file=sys.stderr)
            sys.stderr.flush()
            for i in range(self.n):
                print("{} of {}...".format(i+1, self.n), file=sys.stderr)
                sys.stderr.flush()
                self.alpha_max[i,0] = self.grey_boxes[i].calibrate_scores(Y[i], epsilon=epsilon[i])
        else:
            for i in range(self.n):
                self.alpha_max[i,0] = self.grey_boxes[i].calibrate_scores(Y[i], epsilon=epsilon[i])
                
    def predict(self, X, random_state=2020):
        n = X.shape[0]
        S = [[]]*n
        n_classes = len(self.classes)

        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)
        prop_smaller = np.zeros((n,n_classes))
        
        if self.verbose:
            print("Computing predictive sets for {} samples:". format(n), file=sys.stderr)
            sys.stderr.flush()
            for i in range(self.n):
                print("{} of {}...".format(i+1, self.n), file=sys.stderr)
                sys.stderr.flush()
                gb = ProbAccum(self.mu_LOO[i].predict_proba(X))
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])
        else:
            for i in range(self.n):
                gb = ProbAccum(self.mu_LOO[i].predict_proba(X))
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])
                
        for k in range(n_classes):
            prop_smaller[:,k] /= float(self.n)
        level_adjusted = (1.0-self.alpha)*(1.0+1.0/float(self.n))
        S = [None]*n
        for i in range(n):
            S[i] = np.where(prop_smaller[i,:] < level_adjusted)[0]
            if (not self.allow_empty) and (len(S[i])==0): # Note: avoid returning empty sets
                if len(S[i])==0:
                    S[i] = [np.argmin(prop_smaller[i,:])]            
        return S

class SplitConformal:
    def __init__(self, X, Y, black_box, alpha, random_state=2020, allow_empty=True, verbose=False):
        self.allow_empty = allow_empty

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state)
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

def H(p, eps=.0001):
    return -sum([s * np.log(s + eps) for s in p])

def sigma(t):
    return 1/(np.exp(-t) + 1)

def quantile(v, alpha):
    m = int(np.ceil((1 - alpha) * (len(v) + 1))) - 1
    v = np.sort(v, axis = 0)
    return v[m]

class SplitConformalTransform:
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
    