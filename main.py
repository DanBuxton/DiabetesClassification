import sys
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Modeling Libraries
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, rand_score
from sklearn.model_selection import KFold, GridSearchCV, cross_validate, cross_val_score, StratifiedKFold as SKF, learning_curve, train_test_split

# Machine Learning Libs
from sklearn.ensemble import RandomForestClassifier as RF, AdaBoostClassifier as AB, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC as SVM
from sklearn import metrics, linear_model, datasets
# datasets.load_diabetes()

# My libraries
from MachineLearning import MachineLearning

# # Diabetes Health Indicators Dataset - Kaggle.csv
# # dataset.csv

is_pima_dataset: bool = sys.argv[1]
'''
WARNING: 
False   has 253,680 rows. 
True    has 768     rows. 
'''

if is_pima_dataset:
    dataset = pd.read_csv('./dataset.csv')
    ml = MachineLearning()
    
    # #transform data
    dataset['Pregnancies'] = dataset['Pregnancies'].astype('int32')
    dataset['Glucose'] = dataset['Glucose'].astype('int32')
    dataset['BloodPressure'] = dataset['BloodPressure'].astype('int32')
    dataset['SkinThickness'] = dataset['SkinThickness'].astype('int32')
    dataset['Insulin'] = dataset['Insulin'].astype('float32')
    dataset['BMI'] = dataset['BMI'].astype('float32')
    dataset['DiabetesPedigreeFunction'] = dataset['DiabetesPedigreeFunction'].astype('float32')
    dataset['Age'] = dataset['Age'].astype('int32')
    dataset['Outcome'] = dataset['Outcome'].astype('int32')
else:
    dataset = pd.read_csv('./Diabetes Health Indicators Dataset - Kaggle.csv')
    ml = MachineLearning(dataset='./Diabetes Health Indicators Dataset - Kaggle.csv', label='Diabetes_binary')

    # #transform data
    dataset['Diabetes_binary'] = dataset['Diabetes_binary'].astype('int32')
    dataset['HighBP'] = dataset['HighBP'].astype('int32')
    dataset['HighChol'] = dataset['HighChol'].astype('int32')
    dataset['CholCheck'] = dataset['CholCheck'].astype('int32')
    dataset['BMI'] = dataset['BMI'].astype('float32')
    dataset['Smoker'] = dataset['Smoker'].astype('int32')
    dataset['Stroke'] = dataset['Stroke'].astype('int32')
    dataset['HeartDiseaseorAttack'] = dataset['HeartDiseaseorAttack'].astype('int32')
    dataset['PhysActivity'] = dataset['PhysActivity'].astype('int32')
    dataset['Fruits'] = dataset['Fruits'].astype('int32')
    dataset['Veggies'] = dataset['Veggies'].astype('int32')

    dataset['HvyAlcoholConsump'] = dataset['HvyAlcoholConsump'].astype('int32')
    dataset['AnyHealthcare'] = dataset['AnyHealthcare'].astype('int32')
    dataset['NoDocbcCost'] = dataset['NoDocbcCost'].astype('int32')
    dataset['GenHlth'] = dataset['GenHlth'].astype('int32')
    dataset['MentHlth'] = dataset['MentHlth'].astype('int32')
    dataset['PhysHlth'] = dataset['PhysHlth'].astype('int32')
    dataset['DiffWalk'] = dataset['DiffWalk'].astype('int32')
    dataset['Sex'] = dataset['Sex'].astype('int32')
    dataset['Age'] = dataset['Age'].astype('int32')
    dataset['Education'] = dataset['Education'].astype('int32')
    dataset['Income'] = dataset['Income'].astype('int32')

# ml._df = dataset

# # print(dataset['Age'].describe())

ml.display_info()

for c in ml.columns:
    col = ml.column(c)
    name = col.name
    print(f'\nMean Before {name}: {col.mean()}')
    # print(f'Mode Before {name}: {col.mode()[0]}')
    print(f'Median Before {name}: {col.median()}')
    print(f'Min Before {name}: {col.min()}')
    print(f'Max Before {name}: {col.max()}')

# print(ml.get_column('BMI').describe())

# ml.display_info()
# ml.process_data_and_clean()
# ml.display_info()

print('\n\n')

ml._df = ml._df.drop_duplicates()

for c in ml.columns:
    col = ml.column(c)
    name = col.name
    print(f'\nMean After {name}: {col.mean()}')
    # print(f'Mode After {name}: {col.mode()[0]}')
    print(f'Median After {name}: {col.median()}')
    print(f'Min After {name}: {col.min()}')
    print(f'Max After {name}: {col.max()}')

# print(ml.get_column('BMI').describe())

print('\n')

ml.display_info()

# print(ml.describe_full())

# exit()

import seaborn as sns

plt.figure(figsize=(20,18))
plt.title("Correlation of all attributes")
sns.heatmap(data=dataset.corr(), annot=True, fmt = ".2f", cmap = "coolwarm", vmax=.8)
plt.show()
if is_pima_dataset:
    plt.savefig('PIMA_Heatmap pre-processing')
else:
    plt.savefig('DHI_Heatmap pre-processing')
plt.close()

# for name in dataset.columns:
# for name in dataset.columns:
#     plt.figure()
#     sns.boxplot(x=dataset[name])
#     plt.title("Box Plot before outlier removing")
#     if is_pima_dataset:
#         plt.savefig(f'Figures/PIMA/Before - {name}')
#     else:
#         plt.savefig(f'Figures/DHI/Before - {name}')
#     plt.close()

plt.figure()
sns.boxplot(x=ml.column('BMI'))
plt.title("Box Plot after outlier removing")
if is_pima_dataset:
    plt.savefig(f'Figures/PIMA/Before - BMI')
else:
    plt.savefig(f'Figures/DHI/Before - BMI')
plt.close()

ml.process_data_and_clean()

# Clean
def drop_outliers(dataset, field_name):
    '''
    From https://www.kaggle.com/code/nareshbhat/outlier-the-silent-killer
    '''
    if field_name == 'Outcome' or field_name == 'Diabetes_binary':
        return
    
    iqr = 1.5 * (np.percentile(dataset[field_name], 75) - np.percentile(dataset[field_name], 25))
    dataset.drop(dataset[dataset[field_name] > (iqr + np.percentile(dataset[field_name], 75))].index, inplace=True)
    dataset.drop(dataset[dataset[field_name] < (np.percentile(dataset[field_name], 25) - iqr)].index, inplace=True)

from sklearn.decomposition import PCA

# The estimated noise covariance following the Probabilistic PCA model 
# from Tipping and Bishop 1999. See "Pattern Recognition and Machine Learning" 
# by C. Bishop, 12.2.1 p. 574 or
# http://www.miketipping.com/papers/met-mppca.pdf

# https://tminka.github.io/papers/pca/minka-pca.pdf
# PCA()

# for name in dataset.columns:
#     drop_outliers(dataset, name)
#     plt.figure()
#     sns.boxplot(x=dataset[name])
#     plt.title("Box Plot after outlier removing")
#     if is_pima_dataset:
#         plt.savefig(f'Figures/PIMA/After - {name}')
#     else:
#         plt.savefig(f'Figures/DHI/After - {name}')
#     plt.close()
plt.figure()
sns.boxplot(x=ml.column('BMI'))
plt.title("Box Plot after outlier removing")
if is_pima_dataset:
    plt.savefig(f'Figures/PIMA/After - BMI')
else:
    plt.savefig(f'Figures/DHI/After - BMI')
plt.close()


# from sklearn import metrics



# Data Transformation
q  = QuantileTransformer()
X = q.fit_transform(dataset)
transformeddataset = q.transform(X)
transformeddataset = pd.DataFrame(X)
transformeddataset.columns = dataset.columns

# ml.process_data_and_clean()
# diabetes = datasets.load_diabetes(as_frame=True)
# df: DataFrame = diabetes.target
# print(df.info().to_string())

method = 'pearson' #'spearman'

# plt.figure(figsize=(13,10))
# sns.heatmap(dataset.corr(method=method),annot=True, fmt = ".2f", cmap = "coolwarm", vmin=.2, vmax=.8)
# plt.colorbar()
# if is_pima_dataset:
#     plt.savefig('PIMA_Heatmap post-processing')
# else:
#     plt.savefig('DHI_Heatmap post-processing')
# plt.close();

# ml.process_data_and_clean()

# Data Split
if is_pima_dataset:
    features = transformeddataset.drop(["Outcome"], axis=1)
    label = transformeddataset["Outcome"]
else:
    # Data Split
    features = transformeddataset.drop(["Diabetes_binary"], axis=1)
    label = transformeddataset["Diabetes_binary"]

X_train: Series
X_test: Series
y_train: Series
y_test: Series

X_train, X_test, y_train, y_test =  train_test_split(features, label, test_size=0.30, random_state=7)

if is_pima_dataset:
    joblib_models_file = "joblib_PIMA_models.pkl"
else:
    joblib_models_file = "joblib_DHI_models.pkl"

models = [LR(random_state=30, solver='liblinear')] # joblib.load(joblib_models_file)
model = models[0]
# model.fit(x_train, y_train)

lasso = linear_model.Lasso()
lasso.fit(X_train, y_train)

print(f'selection: {lasso.selection}')

# kfold = SKF(n_splits = 20)
cv_results = cross_validate(estimator = lasso, X = X_train, y = y_train)

# cross_val_score(estimator = model, X = x_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4)

for k in cv_results.keys():
    if str(k).find('time') != -1:
        print(f'{k} {cv_results[k].mean()} second(s)')
    else:
        print(f'{k} {cv_results[k].mean()}')

plt.figure(figsize=(20,18))
plt.title("Correlation of all attributes")
sns.heatmap(data=dataset.corr(), annot=True, fmt = ".2f", cmap = "coolwarm", vmax=.8)
# plt.colorbar()
if is_pima_dataset:
    plt.savefig('PIMA_Heatmap post-processing')
else:
    plt.savefig('DHI_Heatmap post-processing')
plt.show()
plt.close()

# print(transformeddataset.describe())


# print(transformeddataset.corr())


# from sklearn.base import BaseEstimator

# # Modeling step Test differents algorithms 
# random_state = 30
# models: "list[BaseEstimator]" = [
#     LR(random_state = random_state, solver='liblinear'),
#     DT(random_state = random_state),
#     AB(DT(random_state = random_state), random_state = random_state, learning_rate = 0.2),
#     # AB(LR(random_state = random_state, solver='liblinear'), random_state = random_state, learning_rate = 0.2),
#     SVM(random_state = random_state),
#     RF(random_state = random_state),
#     GradientBoostingClassifier(random_state = random_state),
#     KNN()
# ]

# lbfgs (small dataset - fast), sgd, adam (large dataset)
optimisors = ['lbfgs', 'sgd', 'adam']

for optimisor in optimisors:
    models = [
        # [
        # 1 hidden-layers
        MLP(solver=optimisor, max_iter=200, learning_rate_init=0.1, hidden_layer_sizes=5),
        MLP(solver=optimisor, max_iter=200, learning_rate_init=0.01, hidden_layer_sizes=5),
        MLP(solver=optimisor, max_iter=200, learning_rate_init=0.005, hidden_layer_sizes=5),
        MLP(solver=optimisor, max_iter=200, learning_rate_init=0.001, hidden_layer_sizes=5),
        
        MLP(solver=optimisor, max_iter=400, learning_rate_init=0.1, hidden_layer_sizes=5),
        MLP(solver=optimisor, max_iter=400, learning_rate_init=0.01, hidden_layer_sizes=5),
        MLP(solver=optimisor, max_iter=400, learning_rate_init=0.005, hidden_layer_sizes=5),
        MLP(solver=optimisor, max_iter=400, learning_rate_init=0.001, hidden_layer_sizes=5),
        
        MLP(solver=optimisor, max_iter=600, learning_rate_init=0.1, hidden_layer_sizes=5),
        MLP(solver=optimisor, max_iter=600, learning_rate_init=0.01, hidden_layer_sizes=5),
        MLP(solver=optimisor, max_iter=600, learning_rate_init=0.005, hidden_layer_sizes=5),
        MLP(solver=optimisor, max_iter=600, learning_rate_init=0.001, hidden_layer_sizes=5),
        # ],
        # [
        # 2 hidden-layers
        MLP(solver=optimisor, max_iter=200, learning_rate_init=0.1, hidden_layer_sizes=(26, 5)),
        MLP(solver=optimisor, max_iter=200, learning_rate_init=0.01, hidden_layer_sizes=(26, 5)),
        MLP(solver=optimisor, max_iter=200, learning_rate_init=0.005, hidden_layer_sizes=(26, 5)),
        MLP(solver=optimisor, max_iter=200, learning_rate_init=0.001, hidden_layer_sizes=(26, 5)),

        MLP(solver=optimisor, max_iter=400, learning_rate_init=0.1, hidden_layer_sizes=(26, 5)),
        MLP(solver=optimisor, max_iter=400, learning_rate_init=0.01, hidden_layer_sizes=(26, 5)),
        MLP(solver=optimisor, max_iter=400, learning_rate_init=0.005, hidden_layer_sizes=(26, 5)),
        MLP(solver=optimisor, max_iter=400, learning_rate_init=0.001, hidden_layer_sizes=(26, 5)),

        MLP(solver=optimisor, max_iter=600, learning_rate_init=0.1, hidden_layer_sizes=(26, 5)),
        MLP(solver=optimisor, max_iter=600, learning_rate_init=0.01, hidden_layer_sizes=(26, 5)),
        MLP(solver=optimisor, max_iter=600, learning_rate_init=0.005, hidden_layer_sizes=(26, 5)),
        MLP(solver=optimisor, max_iter=600, learning_rate_init=0.001, hidden_layer_sizes=(26, 5)),
        # ],
        # [
        # 3 hidden-layers
        MLP(solver=optimisor, max_iter=200, learning_rate_init=0.1, hidden_layer_sizes=(16, 10, 5)),
        MLP(solver=optimisor, max_iter=200, learning_rate_init=0.01, hidden_layer_sizes=(16, 10, 5)),
        MLP(solver=optimisor, max_iter=200, learning_rate_init=0.005, hidden_layer_sizes=(16, 10, 5)),
        MLP(solver=optimisor, max_iter=200, learning_rate_init=0.001, hidden_layer_sizes=(16, 10, 5)),

        MLP(solver=optimisor, max_iter=400, learning_rate_init=0.1, hidden_layer_sizes=(16, 10, 5)),
        MLP(solver=optimisor, max_iter=400, learning_rate_init=0.01, hidden_layer_sizes=(16, 10, 5)),
        MLP(solver=optimisor, max_iter=400, learning_rate_init=0.005, hidden_layer_sizes=(16, 10, 5)),
        MLP(solver=optimisor, max_iter=400, learning_rate_init=0.001, hidden_layer_sizes=(16, 10, 5)),

        MLP(solver=optimisor, max_iter=600, learning_rate_init=0.1, hidden_layer_sizes=(16, 10, 5)),
        MLP(solver=optimisor, max_iter=600, learning_rate_init=0.01, hidden_layer_sizes=(16, 10, 5)),
        MLP(solver=optimisor, max_iter=600, learning_rate_init=0.005, hidden_layer_sizes=(16, 10, 5)),
        MLP(solver=optimisor, max_iter=600, learning_rate_init=0.001, hidden_layer_sizes=(16, 10, 5))
        # ]
    ]

    # for model in models:
    #     model.fit(x_train, y_train)

    result = []
    score_result = []
    predict_result = []
    matricies = []
    # for m in models:
    for model in models:
        model.fit(X_train, y_train)

        # kfolds = []
        # for i in range(1, 11):
        #     kfolds.append(cross_validate(model, x_train, y_train, cv=1))
        
        # result.append([
        #     cross_validate(model, x_train, y_train, cv=1),
        #     cross_validate(model, x_train, y_train, cv=5),
        #     cross_validate(model, x_train, y_train, cv=10)
        # ])

        # K-Fold 
        result.append(cross_val_score(model, X_train, y_train, cv=10))

        # result.append(cross_val_score(estimator = model, X = x_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

        score_result.append(model.score(X_test, y_test))

        y_pred = model.predict(X_test)
        predict_result.append(y_pred)
        matrix = metrics.confusion_matrix(y_test, y_pred)
        matricies.append(matrix)

        plt.figure(figsize=(8,6))
        sns.heatmap(data=matrix, annot=True, fmt=".0f", cmap='viridis')
        plt.title("Confusion Matrix")
        plt.xlabel("Prediction")
        plt.ylabel("Actual")

        if is_pima_dataset:
            filename = 'Figures/PIMA/CM/'
        else:
            filename = 'Figures/DHI/CM/'
        
        layers:int

        if model.hidden_layer_sizes is tuple:
            layers = model.hidden_layer_sizes.count()
        else:
            layers = model.hidden_layer_sizes

        filename += f'MLP ({optimisor}) Iter_{model.max_iter} Layers_{layers} LearnRate_{model.learning_rate_init}.png'
        plt.savefig(filename)
        plt.close()

    # exit()



    cv_means = []
    cv_std = []
    for cv_result in result:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())
    
    model_result = [
            f"MLP 1-layer ({optimisor}) 200 0.1",
            f"MLP 1-layer ({optimisor}) 200 0.01",
            f"MLP 1-layer ({optimisor}) 200 0.005",
            f"MLP 1-layer ({optimisor}) 200 0.001",

            f"MLP 1-layer ({optimisor}) 400 0.1",
            f"MLP 1-layer ({optimisor}) 400 0.01",
            f"MLP 1-layer ({optimisor}) 400 0.005",
            f"MLP 1-layer ({optimisor}) 400 0.001",

            f"MLP 1-layer ({optimisor}) 600 0.1",
            f"MLP 1-layer ({optimisor}) 600 0.01",
            f"MLP 1-layer ({optimisor}) 600 0.005",
            f"MLP 1-layer ({optimisor}) 600 0.001",



            f"MLP 2-layer ({optimisor}) 200 0.1",
            f"MLP 2-layer ({optimisor}) 200 0.01",
            f"MLP 2-layer ({optimisor}) 200 0.005",
            f"MLP 2-layer ({optimisor}) 200 0.001",

            f"MLP 2-layer ({optimisor}) 400 0.1",
            f"MLP 2-layer ({optimisor}) 400 0.01",
            f"MLP 2-layer ({optimisor}) 400 0.005",
            f"MLP 2-layer ({optimisor}) 400 0.001",

            f"MLP 2-layer ({optimisor}) 600 0.1",
            f"MLP 2-layer ({optimisor}) 600 0.01",
            f"MLP 2-layer ({optimisor}) 600 0.005",
            f"MLP 2-layer ({optimisor}) 600 0.001",



            f"MLP 3-layer ({optimisor}) 200 0.1",
            f"MLP 3-layer ({optimisor}) 200 0.01",
            f"MLP 3-layer ({optimisor}) 200 0.005",
            f"MLP 3-layer ({optimisor}) 200 0.001",

            f"MLP 3-layer ({optimisor}) 400 0.1",
            f"MLP 3-layer ({optimisor}) 400 0.01",
            f"MLP 3-layer ({optimisor}) 400 0.005",
            f"MLP 3-layer ({optimisor}) 400 0.001",

            f"MLP 3-layer ({optimisor}) 600 0.1",
            f"MLP 3-layer ({optimisor}) 600 0.01",
            f"MLP 3-layer ({optimisor}) 600 0.005",
            f"MLP 3-layer ({optimisor}) 600 0.001"
            # "Logistic Regression",
            # "Decision Tree",
            # "Ada Boost (Decision Tree)",
            # # "Ada Boost (LR)",
            # "SVM",
            # "Random Forest",
            # "Gradient Boosting",
            # "KNeighbors"
        ]

    print(len(cv_means))
    print(len(cv_std))
    print(len(score_result))
    print(len(predict_result))
    print(len(model_result))

    result_dataset = pd.DataFrame({
        "CrossVal K10 Accuracy": cv_means,
        "CrossVal K10 errors": cv_std,
        "Score": score_result,
        "Prediction": predict_result,
        "Models": model_result
    })

    # for model in models:
    #     # for model in m:
    #         score_result[model] = model.score(X_test, y_test)
            
    #         y_pred = model.predict(X_test)
    #         predict_result.append(y_pred)
    #         matrices[model] = (metrics.confusion_matrix(y_test, y_pred))
    #         reports[model] = (metrics.classification_report(y_test, y_pred))


            













    plt.figure()

    # Generate chart
    bar = sns.barplot(x = "Score", y = "Models", data = result_dataset, orient = "h")
    bar.set_xlabel("Mean Accuracy")
    bar.set_title("Cross validation scores")

    if is_pima_dataset:
        plt.savefig(f'Figures/PIMA/Score ({optimisor})')
    else:
        plt.savefig(f'Figures/DHI/Score ({optimisor})')
    plt.close()



exit()



# # Data Split
# if is_pima_dataset:
#     features = transformeddataset.drop(["Outcome"], axis=1)
#     labels = transformeddataset["Outcome"]
# else:
#     # Data Split
#     features = transformeddataset.drop(["Diabetes_binary"], axis=1)
#     labels = transformeddataset["Diabetes_binary"]

# X_train, X_test, y_train, y_test = ml.splitDataset(features, labels) # train_test_split(features, labels, test_size=0.30, random_state=7)




# # # ml._dataset = transformeddataset



# # from sklearn.base import BaseEstimator

# # # Modeling step Test differents algorithms 
# # random_state = 30
# # models: "list[BaseEstimator]" = [
# #     LogisticRegression(random_state = random_state, solver='liblinear'),
# #     DecisionTreeClassifier(random_state = random_state),
# #     AdaBoostClassifier(DecisionTreeClassifier(random_state = random_state), random_state = random_state, learning_rate = 0.2),
# #     SVC(random_state = random_state),
# #     RandomForestClassifier(random_state = random_state),
# #     GradientBoostingClassifier(random_state = random_state),
# #     KNeighborsClassifier(),
# # ]

# # for model in models:
# #     model.fit(x_train, y_train)

# def save_models():
#     # from sklearn.externals import joblib

#     import joblib
    
#     if is_pima_dataset:
#         joblib_file = "joblib_PIMA_models.pkl"
#     else:
#         joblib_file = "joblib_DHI_models.pkl"
#     joblib.dump(models, joblib_file)

# if is_pima_dataset:
#     joblib_models_file = "joblib_PIMA_models.pkl"
# else:
#     joblib_models_file = "joblib_DHI_models.pkl"
    
# # models = joblib.load(joblib_file)

# # save_models()

# # exit()

# import joblib
# models = joblib.load(joblib_models_file)

# # # plt.figure(figsize=(16,6))
# # ml.evaluate_model(models, x_train, y_train)

# # Cross validate model with Kfold stratified cross val

# result = []
# score_result = []
# predict_result = []
# for model in models:
#     result.append(cross_val_score(estimator = model, X = X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))
#     # model.fit(x_train, y_train)

#     score_result.append(model.score(X_test, y_test))
    
#     y_pred = model.predict(X_test)
#     predict_result.append(y_pred)

# cv_means = []
# cv_std = []
# for cv_result in result:
#     cv_means.append(cv_result.mean())
#     cv_std.append(cv_result.std())

# result_dataset = pd.DataFrame({
#     "CrossValMeans":cv_means,
#     # "CrossValerrors": cv_std,
#     "Score": score_result,
#     "Prediction": predict_result,
#     "Models":[
#         "Logistic Regression",
#         "Decision Tree",
#         "Decision Tree - AdaBoost",
#         "SVM",
#         "Random Forest",
#         "Gradient Boosting",
#         "KNeighbors"
#     ]
# })

# # plt.figure(figsize=(18,6))

# # # Generate chart
# # bar = sns.barplot(x = "CrossValMeans", y = "Models", data = result_dataset, orient = "h")
# # bar.set_xlabel("Mean Accuracy")
# # bar.set_title("Cross validation scores")

# # if is_pima_dataset:
# #     plt.savefig('PIMA_Results 1')
# # else:
# #     plt.savefig('DHI_Results 1')
# # plt.close()



# plt.figure(figsize=(18,6))

# # Generate chart
# bar = sns.barplot(x = "Score", y = "Models", data = result_dataset, orient = "h")
# bar.set_xlabel("Mean Accuracy")
# bar.set_title("Cross validation scores")

# if is_pima_dataset:
#     plt.savefig('Figures/PIMA/Score')
# else:
#     plt.savefig('Figures/DHI/Score')
# plt.close()



# # plt.figure(figsize=(18,6))

# # # Generate chart
# # bar = sns.barplot(x = "CrossValMeans", y = "Models", data = result_dataset, orient = "h")
# # bar.set_xlabel("Mean Accuracy")
# # bar.set_title("Cross validation scores")

# # if is_pima_dataset:
# #     plt.savefig('Figures/PIMA/Prediction')
# # else:
# #     plt.savefig('Figures/DHI/Prediction')
# # plt.close()















# # for model in models:
# #     score_result[model] = model.score(x_test, y_test)
    
# #     y_pred = model.predict(x_test)
# #     predict_result.append(y_pred)
# #     matrices[model] = (metrics.confusion_matrix(y_test, y_pred))
# #     reports[model] = (metrics.classification_report(y_test, y_pred))


# #     plt.figure(figsize = (8,6))
# #     sns.heatmap(matrices[model], annot = True, fmt = ".0f", cmap = 'viridis')
# #     plt.title("Confusion Matrix")
# #     plt.xlabel("Prediction")
# #     plt.ylabel("Actual")

# #     if is_pima_dataset:
# #         plt.savefig(f'Figures/PIMA/Confusion Matrix ')
# #     else:
# #         plt.savefig(f'Figures/DHI/Confusion Matrix {model}')






# # feature = pd.Series(models[4].feature_importances_, index = x_train.columns).sort_values(ascending = False)

# # plt.figure(figsize = (10,6))
# # sns.barplot(x = feature, y = feature.index)
# # plt.title("Feature Importance")
# # plt.xlabel('Score')
# # plt.ylabel('Features')

# # if is_pima_dataset:
# #     plt.savefig('Figures/PIMA/Score Feature Importance')
# # else:
# #     plt.savefig('Figures/DHI/Score Feature Importance')
# # plt.close()


# # My Transform

# df = ml._df

# #transform data
# if is_pima_dataset:
#     None
# else:
#     df.Diabetes_binary[df['Diabetes_binary'] == 0] = 'No Diabetes'
#     df.Diabetes_binary[df['Diabetes_binary'] == 1] = 'Diabetes'

#     df.HighBP[df['HighBP'] == 0] = 'No High'
#     df.HighBP[df['HighBP'] == 1] = 'High BP'

#     df.HighChol[df['HighChol'] == 0] = 'No High Cholesterol'
#     df.HighChol[df['HighChol'] == 1] = 'High Cholesterol'

#     df.CholCheck[df['CholCheck'] == 0] = 'No Cholesterol Check in 5 Years'
#     df.CholCheck[df['CholCheck'] == 1] = 'Cholesterol Check in 5 Years'

#     df.Smoker[df['Smoker'] == 0] = 'No'
#     df.Smoker[df['Smoker'] == 1] = 'Yes'

#     df.Stroke[df['Stroke'] == 0] = 'No'
#     df.Stroke[df['Stroke'] == 1] = 'Yes'

#     df.HeartDiseaseorAttack[df['HeartDiseaseorAttack'] == 0] = 'No'
#     df.HeartDiseaseorAttack[df['HeartDiseaseorAttack'] == 1] = 'Yes'

#     df.PhysActivity[df['PhysActivity'] == 0] = 'No'
#     df.PhysActivity[df['PhysActivity'] == 1] = 'Yes'

#     df.Fruits[df['Fruits'] == 0] = 'No'
#     df.Fruits[df['Fruits'] == 1] = 'Yes'

#     df.Veggies[df['Veggies'] == 0] = 'No'
#     df.Veggies[df['Veggies'] == 1] = 'Yes'

#     df.HvyAlcoholConsump[df['HvyAlcoholConsump'] == 0] = 'No'
#     df.HvyAlcoholConsump[df['HvyAlcoholConsump'] == 1] = 'Yes'

#     df.AnyHealthcare[df['AnyHealthcare'] == 0] = 'No'
#     df.AnyHealthcare[df['AnyHealthcare'] == 1] = 'Yes'

#     df.NoDocbcCost[df['NoDocbcCost'] == 0] = 'No'
#     df.NoDocbcCost[df['NoDocbcCost'] == 1] = 'Yes'

#     df.GenHlth[df['GenHlth'] == 1] = 'Excellent'
#     df.GenHlth[df['GenHlth'] == 2] = 'Very Good'
#     df.GenHlth[df['GenHlth'] == 3] = 'Good'
#     df.GenHlth[df['GenHlth'] == 4] = 'Fair'
#     df.GenHlth[df['GenHlth'] == 5] = 'Poor'

#     df.DiffWalk[df['DiffWalk'] == 0] = 'No'
#     df.DiffWalk[df['DiffWalk'] == 1] = 'Yes'

#     df.Sex[df['Sex'] == 0] = 'Female'
#     df.Sex[df['Sex'] == 1] = 'Male'

#     df.Age[df['Age'] == 1] = '18-24'
#     df.Age[df['Age'] == 2] = '25-29'
#     df.Age[df['Age'] == 3] = '30-34'
#     df.Age[df['Age'] == 4] = '35-39'
#     df.Age[df['Age'] == 5] = '40-44'
#     df.Age[df['Age'] == 6] = '45-49'
#     df.Age[df['Age'] == 7] = '50-54'
#     df.Age[df['Age'] == 8] = '55-59'
#     df.Age[df['Age'] == 9] = '60-64'
#     df.Age[df['Age'] == 10] = '65-69'
#     df.Age[df['Age'] == 11] = '70-74'
#     df.Age[df['Age'] == 12] = '75-79'
#     df.Age[df['Age'] == 13] = '80 and above'

#     df.Education[df['Education'] == 1] = 'Never Attended School'
#     df.Education[df['Education'] == 2] = 'Elementary'
#     df.Education[df['Education'] == 3] = 'Junior High School'
#     df.Education[df['Education'] == 4] = 'Senior High School'
#     df.Education[df['Education'] == 5] = 'Undergraduate Degree'
#     df.Education[df['Education'] == 6] = 'Magister'
#     # df.Education[df['Education'] == 1] = 'Never Attended School'
#     # df.Education[df['Education'] == 2] = 'Primary School'
#     # df.Education[df['Education'] == 3] = 'Secondary School'
#     # df.Education[df['Education'] == 4] = 'College'
#     # df.Education[df['Education'] == 5] = 'Undergraduate Degree'
#     # df.Education[df['Education'] == 6] = 'Postgraduate Degree'

#     df.Income[df['Income'] == 1] = 'Less Than $10,000'
#     df.Income[df['Income'] == 2] = 'Less Than $10,000'
#     df.Income[df['Income'] == 3] = 'Less Than $10,000'
#     df.Income[df['Income'] == 4] = 'Less Than $10,000'
#     df.Income[df['Income'] == 5] = 'Less Than $35,000'
#     df.Income[df['Income'] == 6] = 'Less Than $35,000'
#     df.Income[df['Income'] == 7] = 'Less Than $35,000'
#     df.Income[df['Income'] == 8] = '$75,000 or More'

# print(df.head())


# exit()

# # print(f'{dataset.describe().to_string()}\n')

# # # plt.figure()
# # # ml.describe()
# # # if is_pima_dataset:
# # #     plt.savefig('PIMA_Describe')
# # # else:
# # #     plt.savefig('DHI_Describe')

# # # print(dataset)
# # # arr = dataset['BloodPressure']

# # x = None
# # y = None

# # if is_pima_dataset:
# #     x = dataset['BloodPressure']
# #     y = dataset['BMI']
# # else:
# #     x = dataset['Education']
# #     y = dataset['Income']

# # # x = dataset['Age']
# # # y = dataset['Pregnancies']

# # # slope, inter, r, p, std_err = stats.linregress(x, y)
# # #
# # # print(r)

# # method = 'spearman'

# # corr = ml.display_corr(method=method)

# # # corr = dataset.corr(method=method) #spearman / pearson / ????? / callable function
# # print(f"{corr.to_string()}\n")

# # # print(dataset.info())

# # # print(x.to_string())

# # # print(f'\n{len(x[x==0])}')

# # # print(dataset.min())
# # # # print(dataset.mean())
# # # print(dataset.mean())



# # # https://medium.com/geekculture/diabetes-prediction-using-machine-learning-python-23fc98125d8

# # # import seaborn as sns

# # plt.figure(figsize=(13,10))
# # sns.heatmap(dataset.corr(method=method), annot=True, fmt = ".2f", cmap = "coolwarm")
# # if is_pima_dataset:
# #     plt.savefig('PIMA_Heatmap before')
# # else:
# #     plt.savefig('DHI_Heatmap before')

# # if is_pima_dataset:
# #     # dataset['Glucose'] = dataset['Glucose'].replace(0, dataset['Glucose'].mean())
# #     # # Correcting missing values in blood pressure
# #     # dataset['BloodPressure'] = dataset['BloodPressure'].replace(0, dataset['BloodPressure'].mean()) # There are 35 records with 0 BloodPressure in dataset
# #     # # Correcting missing values in BMI
# #     # dataset['BMI'] = dataset['BMI'].replace(0, dataset['BMI'].median())
# #     # # Correct missing values in Insulin and SkinThickness

# #     # dataset['SkinThickness'] = dataset['SkinThickness'].replace(0, dataset['SkinThickness'].median())
# #     # dataset['Insulin'] = dataset['Insulin'].replace(0, dataset['Insulin'].median())

# #     ml.process_data_and_clean()


# # plt.figure(figsize=(13,10))
# # sns.heatmap(dataset.corr(method=method),annot=True, fmt = ".2f", cmap = "coolwarm")
# # if is_pima_dataset:
# #     plt.savefig('PIMA_Heatmap')
# # else:
# #     plt.savefig('DHI_Heatmap')

# # if is_pima_dataset:
# #     plt.figure(figsize=(20,10))
# #     sns.scatterplot(data=dataset, x="Glucose", y="BMI", hue="Age", size="Age")
# #     plt.savefig('PIMA_Glucose_BMI_Age')

# #     # Explore Pregnancies vs Outcome
# #     plt.figure(figsize=(13,6))
# #     g = sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"] == 1], 
# #         color="Red", shade = True)
# #     g = sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"] == 0], 
# #         ax =g, color="Green", shade= True)
# #     g.set_xlabel("Pregnancies")
# #     g.set_ylabel("Frequency")
# #     g.legend(["Positive","Negative"])
# #     plt.savefig('PIMA_Pregnancies_Outcome')

# #     # Explore Glucose vs Outcome
# #     plt.figure(figsize=(13,6))
# #     g = sns.kdeplot(dataset["Glucose"][dataset["Outcome"] == 1], color="Red", shade = True)
# #     g = sns.kdeplot(dataset["Glucose"][dataset["Outcome"] == 0], ax =g, color="Green", shade= True)
# #     g.set_xlabel("Glucose")
# #     g.set_ylabel("Frequency")
# #     g.legend(["Positive","Negative"])
# #     plt.savefig('PIMA_Glucose_Outcome')

# # if is_pima_dataset:
# #     # detect outliers from numeric features
# #     outliers_to_drop = ml.detect_outliers(2 ,['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'SkinThickness', 'Insulin', 'Age'])
# # else:
# #     outliers_to_drop = ml.detect_outliers(2 , dataset.columns[:len(dataset.columns)-1])
# # print(f'Outliers: {len(outliers_to_drop)}\n')
# # # print(outliers_to_drop)

# # # dataset.drop(dataset.loc[outliers_to_drop].index, inplace=True)

# # # print(dataset.info())

# # # Mine
# # if is_pima_dataset:
# #     label = dataset['Outcome']
# #     x = dataset['BloodPressure']
# #     y = dataset['BMI']
# # else:
# #     label = dataset['Diabetes_binary']
# #     x = dataset['Education']
# #     y = dataset['Income']

# # slope, intercept, r, p, std_err = stats.linregress(x, y)


# # def myfunc(x):
# #     return slope * x + intercept


# # mymodel = list(map(myfunc, x))

# # # # print(r)

# # if is_pima_dataset:
# #     plt.figure(figsize=(13,6))
# #     plt.scatter(x, y, c=label, cmap='winter')
# #     plt.colorbar()

# #     plt.plot(x, mymodel)

# #     # # plt.clabel('Diabetes', levels=2)
# #     plt.xlabel(x.name)
# #     plt.ylabel(y.name)
# #     plt.title('Diabetes')
# #     plt.savefig('PIMA_BloodPressure_BMI_LinRegress')



# # # https://medium.com/geekculture/diabetes-prediction-using-machine-learning-python-23fc98125d8

# # # Modeling Libraries
# # from sklearn.preprocessing import QuantileTransformer
# # # from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
# # # from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split

# # # Machine Learning Libs
# # from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.neighbors import KNeighborsClassifier
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.svm import SVC

# # # Data Transformation
# # q  = QuantileTransformer()
# # X = q.fit_transform(dataset)
# # transformeddataset = q.transform(X)
# # transformeddataset = pd.DataFrame(X)
# # transformeddataset.columns = dataset.columns

# # # Data Split
# # if is_pima_dataset:
# #     features = dataset.drop(["Outcome"], axis=1)
# #     labels = dataset["Outcome"]
# # else:
# #     # Data Split
# #     features = dataset.drop(["Diabetes_binary"], axis=1)
# #     labels = dataset["Diabetes_binary"]

# # x_train, x_test, y_train, y_test = ml.splitDataset(features, labels) # train_test_split(features, labels, test_size=0.30, random_state=7)

# # # Modeling step Test differents algorithms 
# # random_state = 30
# # models = [
# #     LogisticRegression(random_state = random_state, solver='liblinear'),
# #     DecisionTreeClassifier(random_state = random_state),
# #     AdaBoostClassifier(DecisionTreeClassifier(random_state = random_state), random_state = random_state, learning_rate = 0.2),
# #     SVC(random_state = random_state),
# #     RandomForestClassifier(random_state = random_state),
# #     GradientBoostingClassifier(random_state = random_state),
# #     KNeighborsClassifier(),
# # ]

# # plt.figure(figsize=(16,6))
# # ml.evaluate_model(models, x_train, y_train)

# # if is_pima_dataset:
# #     plt.savefig('PIMA_Results')
# # else:
# #     plt.savefig('DHI_Results')









# # def plot_roc_curve(true_y, y_prob):
# #     """
# #     plots the roc curve based of the probabilities
# #     """

# #     fpr, tpr, thresholds = roc_curve(true_y, y_prob)
# #     plt.plot(fpr, tpr)
# #     plt.xlabel('False Positive Rate')
# #     plt.ylabel('True Positive Rate')

# # from sklearn.metrics import roc_curve

# # fpr, tpr, thresholds = roc_curve(y, y_prob_2)
# # plt.plot(fpr, tpr)

# # plt.show()

# # print(evaluate_model(models))

# # plt.show()

# # dataset.plot()
# # plt.show()

# # plt.xlabel("Blood Pressure")
# # plt.ylabel("Body Mass Index")

# # print(np.var(arr))
# # print(np.std(arr))
# # print(np.mean(arr))

# # import numpy
# # import matplotlib.pyplot as plt
# # numpy.random.seed(2)
# #
# # x = numpy.random.normal(3, 1, 100)
# # y = numpy.random.normal(150, 40, 100) / x
# #
# # train_x = x[:80]
# # train_y = y[:80]
# #
# # test_x = x[80:]
# # test_y = y[80:]
# #
# # mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
# #
# # myline = numpy.linspace(0, 6, 100)
# #
# # plt.scatter(train_x, train_y)
# # plt.plot(myline, mymodel(myline))
# # plt.show()

# # import numpy as np
# # from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
# #
# # n = 10000
# # y = np.array([0] * n + [1] * n)
# # #
# # y_prob_1 = np.array(
# #     np.random.uniform(.25, .5, n//2).tolist() +
# #     np.random.uniform(.3, .7, n).tolist() +
# #     np.random.uniform(.5, .75, n//2).tolist()
# # )
# # y_prob_2 = np.array(
# #     np.random.uniform(0, .4, n//2).tolist() +
# #     np.random.uniform(.3, .7, n).tolist() +
# #     np.random.uniform(.6, 1, n//2).tolist()
# # )

# # import sys
# # import matplotlib
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn import feature_selection
# # from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
# #
# # n = 10000
# # y = np.array([0] * n + [1] * n)
# # #
# # y_prob_1 = np.array(
# #     np.random.uniform(.25, .5, n // 2).tolist() +
# #     np.random.uniform(.3, .7, n).tolist() +
# #     np.random.uniform(.5, .75, n // 2).tolist()
# # )
# # y_prob_2 = np.array(
# #     np.random.uniform(0, .4, n // 2).tolist() +
# #     np.random.uniform(.3, .7, n).tolist() +
# #     np.random.uniform(.6, 1, n // 2).tolist()
# # )
# #
# # def plot_roc_curve(true_y, y_prob):
# #     """
# #     plots the roc curve based of the probabilities
# #     """
# #
# #     fpr, tpr, thresholds = roc_curve(true_y, y_prob)
# #     plt.plot(fpr, tpr)
# #     plt.xlabel('False Positive Rate')
# #     plt.ylabel('True Positive Rate')
# #
# #
# # fpr, tpr, thresholds = roc_curve(y, y_prob_2)
# # plt.plot(fpr, tpr)
# #
# # plt.show()
