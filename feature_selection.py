import pandas as pd
import csv
import time
from nltk.classify import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score
from sklearn.feature_selection import f_classif, chi2, VarianceThreshold, SelectKBest
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from pylab import *
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from skfeature.function.statistical_based import f_score, CFS, chi_square, gini_index, low_variance, t_score
from skfeature.function.similarity_based import reliefF, fisher_score, lap_score, SPEC, trace_ratio
from skfeature.function.information_theoretical_based import CIFE, CMIM, DISR, FCBF, ICAP, JMI, LCSI, MIFS, MIM, MRMR
from skfeature.function.streaming import alpha_investing
from skfeature.function.sparse_learning_based import NDFS, ll_l21, ls_l21, MCFS, RFS, UDFS
from skfeature.function.wrapper import decision_tree_backward, decision_tree_forward, svm_backward, svm_forward
from skfeature.function.structure import graph_fs, group_fs, tree_fs
from sklearn.linear_model import RandomizedLogisticRegression

# INPUT_FILE_NAME = "all_summary.csv"
# INPUT_FILE_NAME = "200classes.csv"
INPUT_FILE_NAME = "probka.csv"

CLASS_NAME = "res_name"

CLASS_NUM = 20

ILLEGAL_ATTRIBUTES = ["pdb_code", "res_id", "chain_id", "local_res_atom_count", "local_res_atom_non_h_count",
                      "local_res_atom_non_h_occupancy_sum", "local_res_atom_non_h_electron_sum",
                      "local_res_atom_non_h_electron_occupancy_sum", "local_res_atom_C_count",
                      "local_res_atom_N_count", "local_res_atom_O_count", "local_res_atom_S_count",
                      "dict_atom_non_h_count", "dict_atom_non_h_electron_sum", "dict_atom_C_count",
                      "dict_atom_N_count", "dict_atom_O_count", "dict_atom_S_count",
                      "fo_col", "fc_col", "weight_col", "grid_space", "solvent_radius",
                      "solvent_opening_radius", "part_step_FoFc_std_min", "part_step_FoFc_std_max",
                      "part_step_FoFc_std_step", "local_volume", "res_coverage", "blob_coverage",
                      "blob_volume_coverage", "blob_volume_coverage_second", "res_volume_coverage",
                      "res_volume_coverage_second", "skeleton_data", "resolution_max_limit", "part_step_FoFc_std_min",
                      "part_step_FoFc_std_max", "part_step_FoFc_std_step"]

classifiers_names = ["Nearest Neighbors",
                     "Random Forest",
                     "Neural Net",
                     "Naive Bayes"
                     ]

classifiers = [
    KNeighborsClassifier(3),
    RandomForestClassifier(n_estimators=10, max_features=10),
    MLPClassifier(alpha=1),
    GaussianNB()
]

scalers_names = [
    # "Standard",
    "Min max"
]

scalers = [
    # StandardScaler(),
    MinMaxScaler()
]

csv_header = [["curr_feature_selection", "curr_parameters", "max_attributes", "curr_attributes", "clf_name", "scl_name",
               "curr_time_fs", "curr_time_clas", "curr_time_total", "accuracy", "macro_recall_score", "macro_precision_score",
               "kappa_score",
               "top5", "top10"]]
curr_feature_selection = ""
curr_parameters = ""
curr_dataset = ""
curr_time_fs = 0
curr_time_clas = 0
curr_time_total = 0
max_attributes = 0
curr_attributes = 0
num_fea = 10


def select_features(function, X, Y, features_num):
    global curr_feature_selection, num_fea
    selector = SelectKBest(function, k=features_num)
    selector.fit(X, Y)
    X = selector.transform(X)
    return X


def top_n_accuracy(y_true, y_proba, top_n=10):
    if y_proba is None:
        return -1
    try:
        top_n_pred = np.argsort(y_proba, axis=1)[:, -top_n:]
        return np.average(
            np.apply_along_axis(np.any, 1, np.equal(top_n_pred, np.repeat(y_true[:, np.newaxis], top_n, 1))))
    except:
        return -1


def cross_validation(X, Y):
    global curr_feature_selection, curr_time_clas
    # iterate over standardization scalers
    for scl_name, scl in zip(scalers_names, scalers):

        # iterate over classifiers
        for clf_name, clf in zip(classifiers_names, classifiers):
            if "SVM" in curr_feature_selection:
                clf = SVC
                clf_name = "SVC"
            if "Decision tree" in curr_feature_selection:
                clf = DecisionTreeClassifier
                clf_name = "Decision tree"
            start2 = time.time()
            clf2 = make_pipeline(scl, clf)
            print('Started classification ' + clf_name + ', scaler: ' + scl_name)
            if clf_name == "Random Forest":
                predicted_probs = cross_val_predict(clf2, X, Y, cv=5, n_jobs=1, method='predict_proba')
            else:
                predicted_probs = cross_val_predict(clf2, X, Y, cv=5, n_jobs=-1, method='predict_proba')
            predicted = np.argsort(predicted_probs, axis=1)[:, -1:]
            print('Finished classification')
            end2 = time.time()
            curr_time_clas = end2 - start2
            provide_metrics_and_results(Y, predicted, predicted_probs, clf_name, scl_name)

            if "SVM" in curr_feature_selection or "Decision tree" in curr_feature_selection:
                break


def provide_metrics_and_results(Y, predicted, predicted_probs, clf_name, scl_name):
    global results, curr_feature_selection, max_attributes, curr_attributes, curr_time_fs, curr_time_clas, curr_time_total, curr_parameters
    precision = precision_score(Y, predicted, average='macro')
    recall = recall_score(Y, predicted, average='macro')
    accuracy = accuracy_score(Y, predicted)
    kappa_score = cohen_kappa_score(Y, predicted)
    top5 = top_n_accuracy(Y, predicted_probs, 5)
    top10 = top_n_accuracy(Y, predicted_probs, 10)

    curr_time_total = curr_time_clas + curr_time_fs
    write_to_csv([curr_feature_selection, curr_parameters, max_attributes, curr_attributes, clf_name, scl_name,
                  curr_time_fs, curr_time_clas, curr_time_total, accuracy, recall, precision, kappa_score,
                  top5, top10])


def select_and_predict(selection_function_name, X, Y, selection_function=None):
    global curr_feature_selection, curr_time_fs, curr_attributes, curr_parameters
    try:
        curr_feature_selection = selection_function_name
        for num_fea in xrange(30, 130, 30):
            print('Started feature selection: ' + curr_feature_selection + ", number of features: " + str(num_fea))
            start1 = time.time()
            if selection_function_name == "Low variance":
                X_after_selection = selection_function(X, num_fea / 130)
                curr_parameters = "threshold 0.5"
            elif selection_function_name == "Randomized Logistic Regression":
                randomized_logistic = RandomizedLogisticRegression(n_jobs=-1, n_resampling=10, sample_fraction=0.5)
                X_after_selection = randomized_logistic.fit(X, Y)
                curr_parameters = "n_resampling 10, sample_fraction 0.5"
            elif selection_function_name == "CFS":
                idx = selection_function(X, Y)
                X_after_selection = X[:, idx[0:num_fea]]
            elif selection_function_name == "Trace ratio":
                idx = selection_function(X, Y, num_fea, style='fisher')
                X_after_selection = X[:, idx[0:num_fea]]
            elif selection_function_name in ["Decision tree forward", "Decision tree backward", "SVM forward",
                                             "SVM tree backward"]:
                idx = selection_function(X, Y, num_fea)
                X_after_selection = X[:, idx]
            elif selection_function_name == "Alpha investing":
                idx = selection_function(X, Y, 0.05, 0.05)
                X_after_selection = X[:, idx]
            elif selection_function_name in ["CIFE", "CMIM", "DISR", "FCBF", "ICAP", "JMI", "LCSI", "MIFS", "MIM",
                                             "MRMR"]:
                idx = selection_function(X, Y, n_selected_features=num_fea)
                X_after_selection = X[:, idx[0:num_fea]]
            else:
                X_after_selection = select_features(selection_function, X, Y, num_fea)
        curr_attributes = X_after_selection.shape[1]
        end1 = time.time()
        print('Finished feature selection')
        curr_time_fs = end1 - start1
        cross_validation(X_after_selection, Y)
    except:
        write_to_csv([curr_feature_selection, curr_parameters, "Error", "", "", "",
                      "", "", "", "", "", "", "",
                      "", ""])


def write_to_csv(row):
    with open("output.csv", "a") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(row)


def main():
    global results, sorted_classes, curr_feature_selection, max_attributes, curr_attributes
    write_to_csv(csv_header)
    print('Started data prepocessing')
    # read data and drop illegal columns
    df = pd.read_csv(INPUT_FILE_NAME, delimiter=";", header=0)
    df.drop(ILLEGAL_ATTRIBUTES, axis=1, inplace=True)
    df = df[((pd.isnull(df[CLASS_NAME])) == False)]

    # change class column to numerical values
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(df[CLASS_NAME]))
    df[CLASS_NAME] = le.transform(df[CLASS_NAME])

    # get numerical data only and change nan to 0
    df = df._get_numeric_data()
    df[np.isnan(df)] = 0

    # select CLASS_NUM most popular classes
    top_classes = df[CLASS_NAME].value_counts().index.tolist()
    df = df[df[CLASS_NAME].isin(top_classes[:CLASS_NUM])]
    sorted_classes = np.sort(np.unique(df[CLASS_NAME]))

    # write df with CLASS_NUM most popular classes to csv
    # df[CLASS_NAME] = le.inverse_transform(df[CLASS_NAME])
    # df.to_csv("200classes.csv", sep=';', encoding='utf-8', index=False)

    # create a numpy array with the numeric values for input into scikit-learn
    numpy_array = df.as_matrix()
    X = numpy_array[:, 1:]
    max_attributes = len(df.columns)
    Y = numpy_array[:, 0]
    for idx, val in enumerate(Y):
        Y[idx] = list(sorted_classes).index(val)
    print('Finished data prepocessing')

    # curr_feature_selection = "Without"
    # curr_attributes = max_attributes
    # cross_validation(X, Y)

    # FEATURE SELECTION

    # STATISTICAL - RFE brakuje (scikit), t_score tylko dla binarnych, chi2 dla kategorycznych i nominalnych

    # select_and_predict("F score", X, Y, f_score.f_score)
    # select_and_predict("Low variance", X, Y, low_variance.low_variance_feature_selection)
    # select_and_predict("Gini index", X, Y, gini_index.gini_index)
    # select_and_predict("Randomized Logistic Regression", X, Y)
    # select_and_predict("CFS", X, Y, CFS.cfs) - dluuugo

    # INFORMATION THEORETICAL BASED - wolniejsze, sprawdzic jedno
    # select_and_predict("CIFE", X, Y, CIFE.cife)
    # select_and_predict("CMIM", X, Y, CMIM.cmim)
    # select_and_predict("DISR", X, Y, DISR.disr)
    # select_and_predict("FCBF", X, Y, FCBF.fcbf)
    # select_and_predict("ICAP", X, Y, ICAP.icap)
    # select_and_predict("JMI", X, Y, JMI.jmi)
    # select_and_predict("LCSI", X, Y, LCSI.lcsi)
    # select_and_predict("MIFS", X, Y, MIFS.mifs)
    # select_and_predict("MIM", X, Y, MIM.mim)
    # select_and_predict("MRMR", X, Y, MRMR.mrmr)

    # SIMILARITY BASED - trace ratio - wszystkie
    # select_and_predict("ReliefF", X, Y, reliefF.reliefF)
    # select_and_predict("Fisher score", X, Y, fisher_score.fisher_score)
    # select_and_predict("Lap score", X, Y, lap_score.lap_score) - samo X
    # select_and_predict("SPEC", X, Y, SPEC.spec) - samo X
    # select_and_predict("Trace ratio", X, Y, trace_ratio.trace_ratio)

    # SPARSE LEARNING BASED - MCFS, NDFS, RFS, UDFS, ll_l21, ls_l21 - sprobowac po jednym

    # STREAMING
    # select_and_predict("Alpha investing", X, Y, alpha_investing.alpha_investing)

    # WRAPPER - decision tree/svm forward/backward
    # select_and_predict("Decision tree forward", X, Y, decision_tree_forward.decision_tree_forward)
    # select_and_predict("Decision tree backward", X, Y, decision_tree_backward.decision_tree_backward)
    # select_and_predict("SVM forward", X, Y, svm_forward.svm_forward)
    # select_and_predict("SVM tree backward", X, Y, svm_backward.svm_backward)

    # STRUCTURE - group, tree, graph feature selections


if __name__ == '__main__':
    main()
