# isort was ran
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from matplotlib.ticker import MultipleLocator, PercentFormatter
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


# path variable to get file
path = Path(__file__).resolve()

# loading numbers into a dictionary
numberFile = loadmat((path.parent / "NumberRecognitionBigger.mat").resolve())

# reshaping data and flattening images for the models
numberFile["X"] = numberFile["X"].transpose(2, 0, 1).reshape(30000, 784)


# helper function for number recognition data
def numberRecogData(matFile):
    # initializing lists for eights and nines, as well as labels
    eightsNines = []
    eightsNinesLabels = []

    # grabbing all eights and nines from dictionary
    for i in range(30000):
        if matFile["y"][0][i] == 8:
            eightsNines.append(matFile["X"][i, :])
            eightsNinesLabels.append(matFile["y"][0][i])
        if matFile["y"][0][i] == 9:
            eightsNines.append(matFile["X"][i, :])
            eightsNinesLabels.append(matFile["y"][0][i])

    # returns numbers and associated labels
    return eightsNines, eightsNinesLabels


def question1(matFile):
    # grabbing data and labels from helper function
    eightsNinesTrain, eightsNinesLabels = numberRecogData(matFile)

    # list of classifiers to loop through
    classifiers = [
        SVC(kernel="linear"),
        SVC(kernel="rbf"),
        RF(n_estimators=100),
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=5),
        KNeighborsClassifier(n_neighbors=10),
    ]

    # list of classifier's names
    classifiersLabeled = ["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"]

    # initializing 2D lists of error rates, each interior list represents each model
    bestErrorRate = [[], [], [], [], [], []]
    meanErrorRate = [[], [], [], [], [], []]
    worstErrorRate = [[], [], [], [], [], []]

    # initializing dataframe of kfold_scores
    kfold_scores = pd.DataFrame(columns=["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"], index=["err"])
    kf_score = []

    # couldn't overwrite NaN values in the df, this fixes it
    kfold_scores = kfold_scores.fillna(0.0)

    # initializing dictionary to make graphing easier
    valuesDict = {
        "best": [0, 0, 0, 0, 0, 0],
        "mean": [0, 0, 0, 0, 0, 0],
        "worst": [0, 0, 0, 0, 0, 0],
    }

    # looping through each classifier, fitting them to training data,
    # cross-validating with the Stratified K-Fold model with 5 folds
    # and scoring with accuracy, then grabbing mean
    # and best score
    for i in range(len(classifiers)):
        # setting current classifier from list
        curClassifier = classifiers[i]
        # initializing Stratified K-Fold model
        StratKFold = StratifiedKFold(5, shuffle=True, random_state=40)

        # performing cross validation using sklearn's cross_validate, with
        # cv parameter set to the stratified k-fold model
        crossVal = cross_validate(
            curClassifier,
            eightsNinesTrain,
            eightsNinesLabels,
            cv=StratKFold,
            scoring="accuracy",
        )

        # getting highest scoring fold for each model
        bestScore = max(crossVal["test_score"])
        # getting lowest scoring fold for each model
        worstScore = min(crossVal["test_score"])

        # adding each respective error rate to array and dict
        bestErrorRate[i].append(round(1 - (bestScore), 4))
        meanErrorRate[i].append(round(1 - (crossVal["test_score"].mean()), 4))
        worstErrorRate[i].append(round(1 - (worstScore), 4))
        valuesDict["best"][i] = min(bestErrorRate[i]) * 100
        valuesDict["mean"][i] = np.mean(meanErrorRate[i]) * 100
        valuesDict["worst"][i] = max(worstErrorRate[i]) * 100

        # getting mean score to put in table for current classifier
        kf_score = np.mean(meanErrorRate[i])
        kfold_scores.loc["err", classifiersLabeled[i]] = kf_score

    sbn.set_style(style="darkgrid")

    # found code to make a faceted-esque bar plot for various models
    # using matplotlib. idea adapted from:
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    xAxis = np.arange(len(bestErrorRate))
    width = 0.25
    multiplier = 0

    fig, axis = plt.subplots(layout="constrained")

    # turning dict into list for easier access
    errors = list(valuesDict.items())
    for error, values in errors:
        offset = width * multiplier
        rects = axis.bar(xAxis + offset, values, width, label=error)
        axis.bar_label(rects, padding=3)
        multiplier += 1

    axis.set_ylabel("Error Rates")
    axis.set_title("Best, Mean and Worst Error Rates for Models (5 folds)")
    axis.set_xticks(xAxis + width, classifiersLabeled)
    axis.legend(loc="upper left", ncols=3)
    axis.set_ylim(0, 5)

    plt.gca().yaxis.set_major_formatter(PercentFormatter())  # set y axis to show percents
    plt.tight_layout()  # so all tick labels are shown properly
    plt.savefig("bonus1.png")
    # clear figure so no aesthetics get carried over to future plots
    plt.clf()

    # provided function to save K-Fold scores to json file from a dataframe
    def save_mnist_kfold(kfold_scores: pd.DataFrame) -> None:
        from pathlib import Path

        import numpy as np
        from pandas import DataFrame

        COLS = sorted(["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"])
        df = kfold_scores
        if not isinstance(df, DataFrame):
            raise ValueError("Argument `kfold_scores` to `save` must be a pandas DataFrame.")
        if kfold_scores.shape != (1, 6):
            raise ValueError("DataFrame must have 1 row and 6 columns.")
        if not np.all(sorted(df.columns) == COLS):
            raise ValueError("Columns are incorrectly named.")
        if not df.index.values[0] == "err":
            raise ValueError("Row has bad index name. Use `kfold_score.index = ['err']` to fix.")

        if np.min(df.values) < 0 or np.max(df.values) > 0.10:
            raise ValueError(
                "Your K-Fold error rates are too extreme. Ensure they are the raw error rates,\r\n"
                "and NOT percentage error rates. Also ensure your DataFrame contains error rates,\r\n"
                "and not accuracies. If you are sure you have not made either of the above mistakes,\r\n"
                "there is probably something else wrong with your code. Contact the TA for help.\r\n"
            )

        if df.loc["err", "svm_linear"] > 0.07:
            raise ValueError("Your svm_linear error rate is too high. There is likely an error in your code.")
        if df.loc["err", "svm_rbf"] > 0.03:
            raise ValueError("Your svm_rbf error rate is too high. There is likely an error in your code.")
        if df.loc["err", "rf"] > 0.05:
            raise ValueError("Your Random Forest error rate is too high. There is likely an error in your code.")
        if df.loc["err", ["knn1", "knn5", "knn10"]].min() > 0.04:
            raise ValueError("One of your KNN error rates is too high. There is likely an error in your code.")

        outfile = Path(__file__).resolve().parent / "kfold_mnist.json"
        df.to_json(outfile)
        print(f"K-Fold error rates for MNIST data successfully saved to {outfile}")

    save_mnist_kfold(kfold_scores)


# loading csv into dataframe for q2 and q3 using pandas
breastCancerDF = pd.read_csv((path.parent / "breast-cancer.csv.xls").resolve())


# helper fuction for accessing the breast cancer dataframe's splits
def breastCancerData(df):
    # cleaning up data
    df = df.drop(["id"], axis=1)  # don't need id values

    # representing the labels numerically for the models
    df["diagnosis"] = df["diagnosis"].replace({"M": 1, "B": 0})
    # splitting df into just the features and just the labels
    testingData = df.drop(["diagnosis"], axis=1)
    labels = df["diagnosis"]

    # normalizing data in between 0 and 1
    # using sklearn's MinMaxScaler. adapted from:
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

    # looping through dataframe to normalize each column
    for i in testingData:
        testingData[i] = MinMaxScaler().fit_transform(testingData[i].values.reshape(-1, 1))

    return testingData, labels


def question2(bcFile):
    # grabbing data from helper function
    testingData, labels = breastCancerData(bcFile)

    # all feature names for breast cancer dataset
    FEAT_NAMES = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave_points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave_points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave_points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ]
    # column names in dataframe
    COLS = [
        "Feature",
        "AUC",
    ]
    # initialized dataframe for AUC values
    aucs = pd.DataFrame(
        columns=COLS,
        data=np.zeros([len(FEAT_NAMES), len(COLS)]),
    )

    for i, feat_name in enumerate(FEAT_NAMES):
        auc = roc_auc_score(y_true=labels, y_score=testingData.iloc[:, i])
        # fixing AUC values that are below 0.5
        if auc < 0.5:
            auc = 1 - auc
        aucs.iloc[i] = (feat_name, auc)

    # sorting by AUC values in descending order,
    # resetting the indicies to match the positions of the values
    aucs_sorted = aucs.sort_values(by="AUC", ascending=False).reset_index(drop=True)
    pd.DataFrame.to_json(aucs_sorted, (Path(__file__).resolve().parent / "aucs.json"))

    # grabbing top ten auc values from the dataframe
    topTenAUC = aucs_sorted.head(10)

    # setting size of figure
    plt.figure(figsize=(8, 6))
    plt.xlabel("AUC Scores")
    plt.tick_params(axis="y", which="major", labelsize=10)
    sbn.barplot(y=topTenAUC["Feature"], x=topTenAUC["AUC"], orient="horizontal").set(
        title="Ten Most Important Features shown through AUC values"
    )
    plt.xlim(0.75, 1.0)
    plt.xticks(np.arange(0.75, 1.0, 0.05))  # make AUC values tick up by 0.05
    plt.gca().set_axisbelow(True)  # make gridlines appear under bars
    plt.grid()
    plt.tight_layout()  # so the full feature names are shown
    plt.savefig("bonus2.png")
    # clear figure so no aesthetics get carried over to future plots
    plt.clf()


def question3(breastCancerDF):
    # grabbing data and labels from helper function
    testingData, labels = breastCancerData(breastCancerDF)

    # just like question 1, initialized list of classifiers to loop through
    classifiers = [
        SVC(kernel="linear"),
        SVC(kernel="rbf"),
        RF(n_estimators=100),
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=5),
        KNeighborsClassifier(n_neighbors=10),
    ]

    # and list of classifier's names
    classifiersLabeled = ["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"]

    # dictionary to store AUC values over each fold for each model
    testDataDict = {}

    # initializing 2D lists of error rates, each interior list represents each model
    bestErrorRate = [[], [], [], [], [], []]
    meanErrorRate = [[], [], [], [], [], []]
    worstErrorRate = [[], [], [], [], [], []]

    # initializing dataframe of kfold_scores
    kfold_scores = pd.DataFrame(columns=["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"], index=["err"])
    kf_score = []

    # couldn't overwrite NaN values in the df, this fixes it
    kfold_scores = kfold_scores.fillna(0.0)

    # initializing dictionary to make graphing easier
    valuesDict = {
        "best": [0, 0, 0, 0, 0, 0],
        "mean": [0, 0, 0, 0, 0, 0],
        "worst": [0, 0, 0, 0, 0, 0],
    }

    for i in range(len(classifiers)):
        # grabbing current classifier from list
        curClassifier = classifiers[i]

        # using Stratified Shuffle Split with 5 folds this time rather
        # than Stratified K-Fold for cross validation. Very similar cross validators anyway
        stratShuffleSplit = StratifiedShuffleSplit(5, test_size=0.2, random_state=40)

        # performing cross validation using sklearn's cross_validate, with
        # cv parameter set to the stratified shuffle split model
        crossVal = cross_validate(
            curClassifier,
            testingData,
            labels,
            cv=stratShuffleSplit,
            scoring=["accuracy", "roc_auc"],
        )

        testDataDict[classifiersLabeled[i]] = list(crossVal["test_roc_auc"])

        # getting highest scoring fold for each model
        bestScore = max(crossVal["test_accuracy"])
        # getting lowest scoring fold for each model
        worstScore = min(crossVal["test_accuracy"])

        # adding each respective error rate to array and dict
        bestErrorRate[i].append(round(1 - (bestScore), 4))
        meanErrorRate[i].append(round(1 - (crossVal["test_accuracy"].mean()), 4))
        worstErrorRate[i].append(round(1 - (worstScore), 4))
        valuesDict["best"][i] = min(bestErrorRate[i]) * 100
        valuesDict["mean"][i] = np.mean(meanErrorRate[i]) * 100
        valuesDict["worst"][i] = max(worstErrorRate[i]) * 100

        # getting mean error rate to put in table for current classifier
        kf_score = np.mean(meanErrorRate[i])
        kfold_scores.loc["err", classifiersLabeled[i]] = kf_score

    # kfold dataframe to json function provided
    def save_data_kfold(kfold_scores: pd.DataFrame) -> None:
        from pathlib import Path

        import numpy as np
        from pandas import DataFrame

        COLS = sorted(["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"])
        df = kfold_scores
        if not isinstance(df, DataFrame):
            raise ValueError("Argument `kfold_scores` to `save` must be a pandas DataFrame.")
        if kfold_scores.shape != (1, 6):
            raise ValueError("DataFrame must have 1 row and 6 columns.")
        if not np.all(sorted(df.columns) == COLS):
            raise ValueError("Columns are incorrectly named.")
        if not df.index.values[0] == "err":
            raise ValueError("Row has bad index name. Use `kfold_score.index = ['err']` to fix.")

        outfile = Path(__file__).resolve().parent / "kfold_data.json"
        df.to_json(outfile)
        print(f"K-Fold error rates for individual dataset successfully saved to {outfile}")

    save_data_kfold(kfold_scores)

    # setting plots styles to seaborn's darkgrid
    sbn.set_style(style="darkgrid")

    # adapted code from question 1 to make facet-esque barplot of best,
    # mean and worst error rates for models
    xAxis = np.arange(len(bestErrorRate))
    width = 0.25
    multiplier = 0

    fig, axis = plt.subplots(layout="constrained")

    # turning dict into list for easier access
    errors = list(valuesDict.items())
    for error, values in errors:
        offset = width * multiplier
        rects = axis.bar(xAxis + offset, values, width, label=error)
        axis.bar_label(rects, padding=3)
        multiplier += 1

    axis.set_ylabel("Error Rates")
    axis.set_title("Best, Mean and Worst Error Rates for Models (5 folds)")
    axis.set_xticks(xAxis + width, classifiersLabeled)
    axis.legend(loc="upper left", ncols=3)
    axis.set_ylim(0, 10)

    plt.gca().yaxis.set_major_formatter(PercentFormatter())  # set y axis to show percents
    plt.tight_layout()  # so all tick labels are shown properly
    plt.savefig("bonus3.png")
    # clear figure so no aesthetics get carried over to future plots
    plt.clf()

    # setting plots styles to seaborn's darkgrid
    sbn.set_style(style="darkgrid")

    # setting figure size
    plt.figure(figsize=(10, 6))

    # looping through arrays inside the dictionary and plotting each models line
    for label, values in testDataDict.items():
        plt.plot(range(1, len(values) + 1), values, marker="o", label=label)

    # setting tick locations and values using MultipleLocator
    x_locator = MultipleLocator(base=1.0)
    plt.gca().xaxis.set_major_locator(x_locator)
    y_locator = MultipleLocator(base=0.01)
    plt.gca().yaxis.set_major_locator(y_locator)

    plt.xlabel("Fold")
    plt.ylabel("AUC Score")
    plt.title("AUC Values Over Folds")
    plt.legend(loc="lower left", ncols=5)
    plt.savefig("bonus4.png")
    # clear figure so no aesthetics get carried over to future plots
    plt.clf()


def bonus(matFile, breastCancerDF):
    # once again, I was curious of the accuracy, precision,
    # recall and F1 scores for these tests.
    # Code adapted from assignment #1

    testingData, labels = breastCancerData(breastCancerDF)
    eightsNinesTrain, eightsNinesLabels = numberRecogData(matFile)

    # had to change labels from 8's and 9's to 0's and 1's because
    # the cross_validate functions recall, precision and F1 scores require
    # positive labels to be 1 and negatives to be 0
    for i in range(len(eightsNinesLabels)):
        if eightsNinesLabels[i] == 8:
            eightsNinesLabels[i] = 1
        if eightsNinesLabels[i] == 9:
            eightsNinesLabels[i] = 0

    classifiers = [
        SVC(kernel="linear"),
        SVC(kernel="rbf"),
        RF(n_estimators=100),
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=5),
        KNeighborsClassifier(n_neighbors=10),
    ]

    classifiersLabeled = ["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"]

    # initializing Stratified K-Fold model for cross validation
    StratKFold = StratifiedKFold(5, shuffle=True, random_state=40)

    # initializing dataframes for accuracy, precision, recall and F1 scores
    # for the breast cancer and number recognition datasets
    bcScoresDF = pd.DataFrame(
        columns=["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"],
        index=["accuracy", "precision", "recall", "F1"],
    )

    numsScoresDF = pd.DataFrame(
        columns=["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"],
        index=["accuracy", "precision", "recall", "F1"],
    )

    # looping through both datasets and testing each classifier through
    # cross-validating with the Stratified K-Fold model with 5 folds
    for i in range(len(classifiers)):
        curClassifier = classifiers[i]

        crossValNum = cross_validate(
            curClassifier,
            eightsNinesTrain,
            eightsNinesLabels,
            cv=StratKFold,
            scoring=["accuracy", "recall", "precision", "f1"],
        )
        # adding mean scores for number recognition from 5 folds to dataframe
        numsScoresDF.loc["accuracy", classifiersLabeled[i]] = np.mean(crossValNum["test_accuracy"]).round(3)
        numsScoresDF.loc["recall", classifiersLabeled[i]] = np.mean(crossValNum["test_recall"]).round(3)
        numsScoresDF.loc["precision", classifiersLabeled[i]] = np.mean(crossValNum["test_precision"]).round(3)
        numsScoresDF.loc["F1", classifiersLabeled[i]] = np.mean(crossValNum["test_f1"]).round(3)

        crossValCancer = cross_validate(
            curClassifier, testingData, labels, cv=StratKFold, scoring=["accuracy", "recall", "precision", "f1"]
        )
        # adding mean scores for cancer prediction from 5 folds to dataframe
        bcScoresDF.loc["accuracy", classifiersLabeled[i]] = np.mean(crossValCancer["test_accuracy"]).round(3)
        bcScoresDF.loc["recall", classifiersLabeled[i]] = np.mean(crossValCancer["test_recall"]).round(3)
        bcScoresDF.loc["precision", classifiersLabeled[i]] = np.mean(crossValCancer["test_precision"]).round(3)
        bcScoresDF.loc["F1", classifiersLabeled[i]] = np.mean(crossValCancer["test_f1"]).round(3)


def main():
    question1(numberFile)
    question2(breastCancerDF)
    question3(breastCancerDF)
    bonus(numberFile, breastCancerDF)


if __name__ == "__main__":
    main()
