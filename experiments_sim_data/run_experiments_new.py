import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import os.path
from os import path

import sys
sys.path.insert(0, '..')
import arc

# Where to write results
out_dir = "/Users/lorry/Documents/20240125 conformal prediction/arc_new/arc/results"

def assess_predictions(S, X, y):
    # Marginal coverage
    coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    # Average length
    length = np.mean([len(S[i]) for i in range(len(y))])
    # Average length conditional on coverage
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    length_cover = np.mean([len(S[i]) for i in idx_cover])
    # Conditional coverage (WSC)
    cond_coverage = arc.coverage.wsc_unbiased(X, y, S)
    # F1 score
    def F1CPscore(sets, y):
        mclasses = len(set(y))
        TP, FP, TN, FN = [[0 for m in range(mclasses)] for k in [0, 1, 2, 3]]
        for i in range(len(sets)):
            interval = sets[i]
            positives = [j for j in range(mclasses) if j in interval]
            negatives = [j for j in range(mclasses) if j not in interval]
            for m in range(mclasses):
                TP[m] = TP[m] + 1 * (m in positives) * (y[i] == m)# in positives)
                FP[m] = FP[m] + 1 * (m in positives) * (y[i] != m)# in negatives)
                TN[m] = TN[m] + 1 * (m in negatives) * (y[i] !=m)# in negatives) 
                FN[m] = FN[m] + 1 * (m in negatives) * (y[i] == m)# in positives) 
        F1cp = np.prod([2 * TP[m]/(2 * TP[m] + FP[m] + FN[m]) for m in  range(mclasses)])
        return F1cp
    def F1score(sets, y):
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
    F1 = F1score(S, y)
    # Combine results
    out = pd.DataFrame({'Coverage': [coverage], 'Conditional coverage': [cond_coverage],
                        'Length': [length], 'Length cover': [length_cover], 
                        'F1': [F1]})
    return out

def collect_predictions(S, X, y, condition_on):
    cover = np.array([y[i] in S[i] for i in range(len(y))])
    length = np.array([len(S[i]) for i in range(len(y))])
    out = pd.DataFrame({'Cover': cover, 'Length': length})
    for j in condition_on:
        var_name = "X{}".format(j)
        out[var_name] = X[:,j]
    return out

def run_experiment(data_model, n_train, methods, black_boxes, condition_on,
                   alpha=0.1, experiment=0, random_state=2020, out_files=[]):
    # Set random seed
    np.random.seed(random_state)

    # Total number of samples
    n_test = 5000
    n = n_train + n_test

    # Generate data
    X = data_model.sample_X(n)
    y = data_model.sample_Y(X)
    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=random_state)

    # Load pre-computed results
    if path.exists(out_files[0]) & path.exists(out_files[1]):
        results = pd.read_csv(out_files[0])
        results_full = pd.read_csv(out_files[1])
    else:
        results = pd.DataFrame()
        results_full = pd.DataFrame()
        
    for box_name in black_boxes:
        black_box = black_boxes[box_name]
        for method_name in methods:
            # Skip if this experiment has already been run
            if results.shape[0] > 0:
                found  = (results['Method']==method_name)
                found &= (results['Black box']==box_name)
                found &= (results['Experiment']==experiment)
                found &= (results['Nominal']==1-alpha)
                found &= (results['n_train']==n_train)
                found &= (results['n_test']==n_test)
            else:
                found = 0
            if np.sum(found) > 0:
                print("Skipping experiment with black-box {} and method {}...".format(box_name, method_name))
                sys.stdout.flush()
                continue

            print("Running experiment with black-box {} and method {}...".format(box_name, method_name))
            sys.stdout.flush()

            # Train classification method
            try:
                method = methods[method_name](X_train, y_train, black_box, alpha, random_state=random_state,
                                              verbose=True, box_name=box_name)
            except:
                method = methods[method_name][0](X_train, y_train, black_box, alpha, random_state=random_state,
                                              model=methods[method_name][1], verbose=True, box_name=box_name)
            # Apply classification method
            S = method.predict(X_test)

            # Evaluate results
            res = assess_predictions(S, X_test, y_test)
            # Add information about this experiment
            res['Method'] = method_name
            res['Black box'] = box_name
            res['Experiment'] = experiment
            res['Nominal'] = 1-alpha
            res['n_train'] = n_train
            res['n_test'] = n_test

            # Evaluate results (conditional)
            res_full = collect_predictions(S, X_test, y_test, condition_on)
            # Add information about this experiment
            res_full['Method'] = method_name
            res_full['Black box'] = box_name
            res_full['Experiment'] = experiment
            res_full['Nominal'] = 1-alpha
            res_full['n_train'] = n_train
            res_full['n_test'] = n_test

            # Add results to the list
            results = pd.concat([results, res]) # results.append(res)
            results_full = pd.concat([results_full, res]) # results_full.append(res_full)

            # Write results on output files
            if len(out_files) == 2:
                results.to_csv(out_files[0], index=False, float_format="%.4f")
                print("Updated summary of results on\n {}".format(out_files[0]))
                results_full.to_csv(out_files[1], index=False, float_format="%.4f")
                print("Updated full results on\n {}".format(out_files[1]))
                sys.stdout.flush()

    return results, results_full

if __name__ == '__main__':
    # Parse input arguments
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    model_num = 1
    out_dir = "/Users/lorry/Documents/20240125 conformal prediction/arc_new/arc/results"
    import itertools
    for model_num, exp_num, alpha, n_train in itertools.product([0], np.arange(5), [0.1], [10000]):
        # sys.argv = [out_dir, 0, 0, 0.1, 10000]
        # if len(sys.argv) != 5:
        #     quit()
        # model_num = int(sys.argv[1])
        # exp_num = int(sys.argv[2])
        # alpha = float(sys.argv[3])
        # n_train = int(sys.argv[4])

        # Determine output file
        out_file_1 = "{}/model{}_exp{}_alpha{}_n{}_summary.txt".format(out_dir, model_num, exp_num, alpha, n_train)
        out_file_2 = "{}/model{}_exp{}_alpha{}_n{}_full.txt".format(out_dir, model_num, exp_num, alpha, n_train)
        out_files = [out_file_1, out_file_2]
        print(out_files)

        # Random state for this experiment
        random_state = 2020 + exp_num

        # Define data model
        np.random.seed(random_state)
        if model_num == 1:
            K = 10
            p = 10
            data_model = arc.models.Model_Ex1(K,p)
        else:
            K = 4
            p = 5
            data_model = arc.models.Model_Ex2(K,p)

        # List of calibration methods to be compared
        methods = {
    #         'ARC':  arc.methods.SplitConformal,
    #         'LRC(HR)': (arc.methods.SplitConformalTransform, 'HR'),
            'LRC(HR)': (arc.methods.SplitConformalTransform, 'ER'),
    #         'LR': arc.methods.SplitConformalEntropy,
            'LRC(base)': (arc.methods.SplitConformalTransform, 'unweighted'),
    #         'HCC': arc.methods.SplitConformalHomogeneous,
    #         'CQC': arc.others.CQC
        }

        # List of black boxes to be compared
        black_boxes = {
    #                    'Oracle': arc.black_boxes.Oracle(data_model),
                       'SVC': arc.black_boxes.SVC(clip_proba_factor = 1e-5, random_state=random_state),
                       'RFC': arc.black_boxes.RFC(clip_proba_factor = 1e-5,
                                                  n_estimators=1000, max_depth=5, max_features=None,
                                                  random_state=random_state)
                      }

        # Which special variables should we condition on to compute conditional coverage?
        condition_on = [0]

        # Run experiments
        run_experiment(data_model, n_train, methods, black_boxes, condition_on,
                       alpha=alpha, experiment=exp_num, random_state=random_state,
                       out_files=out_files)
