from typing import final
import pandas as pd
from pandas import(
    DataFrame,
    Series
)
import seaborn as sns

from sklearn.base import BaseEstimator
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

@final
class MachineLearning:
    '''
    Functionality based on https://medium.com/geekculture/diabetes-prediction-using-machine-learning-python-23fc98125d8
    '''

    _df: DataFrame = None
    _label: Series = None

    def __init__(self, dataset: str='./dataset.csv', label: str = 'Outcome'):
        data = pd.read_csv(dataset)
        self._df = data
        self._label = data[label]
    
    def _col(self, col: str):
        return self._df[col];
    @property
    def _cols(self):
        return self._df.columns;

    @final
    def column(self, col: str):
        return self._col(col)

    @property
    @final
    def columns(self):
        return self._cols

    @final
    def display_info(self):
        return self._df.info()

    @final
    def description(self, attr: str = None, percentiles=[], include=None, exclude=None):
        if attr != None:
            return self._df[attr].describe(percentiles=percentiles, include=include, exclude=exclude)
        else:
            return self._df.describe(percentiles=percentiles, include=include, exclude=exclude)
    @property
    @final
    def full_description(self):
        return self.description().to_string()
    @final
    def display_corr(self, method: str = 'pearson'):
        return self._df.corr(method=method)
    
    # @final
    def process_data_and_clean(self):
        '''
        Cleans and processes the dataset making it ready for modeling
        '''

        self._df = self._df.drop_duplicates()

        # Correct missing values
        for c in self._cols:
            col = self._df[c]

            if col.name != self._label.name and col.name != 'Pregnancies':
                self._df[c] = col.replace(0, col.mean())
        
        # cols = self._cols.to_list()
        # # cols.remove(self._label.index)
        # self._df.drop(self._df.loc[self.detect_outliers(0, cols)].index, inplace=True)

        # # self._df = df

        self._df = self._df.drop_duplicates()

        return None
    @final
    def detect_outliers(self, n: int, features):
        """
        Detect outliers from given list of features. It returns a list of the indices
        according to the observations containing more than n outliers according
        to the Tukey method
        """

        import numpy as np
        from collections import Counter

        outlier_indices = []
        # iterate over features(columns)
        for col in features:
            Q1 = np.percentile(self._df[col], 25)
            Q3 = np.percentile(self._df[col], 75)
            IQR = Q3 - Q1
            
            # outlier step
            outlier_step = 1.5 * IQR
            
            # Determine a list of indices of outliers for feature col
            outlier_list_col = self._df[(self._df[col] < Q1 - outlier_step) | (self._df[col] > Q3 + outlier_step )].index
            
            # append the found outlier indices for col to the list of outlier indices 
            outlier_indices.extend(outlier_list_col)
            
        # select observations containing more than n outliers
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
        
        return multiple_outliers   
    
    def evaluate_models(self, models: "list[BaseEstimator]", x_train, y_train):
        """
        Takes a list of models and returns chart of cross validation scores using mean accuracy
        """
        
        # Cross validate model with Kfold stratified cross val
        kfold = StratifiedKFold(n_splits = 10)
        
        result = []
        for model in models:
            result.append(cross_val_score(estimator = model, X = x_train, y = y_train, scoring = "accuracy", cv = 10, n_jobs=4))

        cv_means = []
        cv_std = []
        for cv_result in result:
            cv_means.append(cv_result.mean())
            cv_std.append(cv_result.std())

        result_dataset = pd.DataFrame({
            "CrossValMeans":cv_means,
            "CrossValerrors": cv_std,
            "Models":[
                "LogisticRegression",
                "DecisionTreeClassifier",
                "AdaBoostClassifier",
                "SVC",
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "KNeighborsClassifier"
            ]
        })

        # Generate chart
        bar = sns.barplot(x = "CrossValMeans", y = "Models", data = result_dataset, orient = "h")
        bar.set_xlabel("Mean Accuracy")
        bar.set_title("Cross validation scores")
        return result_dataset
    @final
    def splitDataset(self, features, labels, test_size=0.30, random_state=7):
        return train_test_split(features, labels, test_size=test_size, random_state=random_state)
    
    