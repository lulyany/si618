# %% [markdown]
# # SI 618 - Homework #7: Classifiers
# or: How I Learned to Stop Worrying and Love Machine Learning
#
# Version 2023.11.08.1.CT

# %% [markdown]
# This is, perhaps, one of the most exciting homework assignments that you have encountered in this course!
#
# You are going to try your hand at a Kaggle competition to predict which passengers on board the Spaceship Titanic are transported to an alternate dimension.
#
# You can access the competition here: **https://www.kaggle.com/c/spaceship-titanic**
#
# This assignment is similar to the Kaggle competition that we did in class, but it uses a different and larger dataset.
#
# The basic steps for this assignment are the same as what we did in class:
#
# 1. Accept the rules and join the competition
# 2. Download the data (from the data tab of the competition page)
# 3. Understand the problem
# 4. EDA (Exploratory Data Analysis)
# 5. Train, tune, and ensemble (!) your machine learning models
# 6. Upload your prediction as a submission on Kaggle and receive an accuracy score
#
# additionally, you will
#
# 7. Upload your final notebook to Canvas and report your best accuracy score.
#
# Note that class grades are not entirely dependent on your accuracy score.
# All models that achieve 80% accuracy will receive full points for
# the accuracy component of this assignment.
#
# Rubric:
#
# 1. (20 points) Conduct an EDA. You must demonstrate that you understand the data and the problem.
# 2. (60 points) Train, tune, and ensemble machine learning models.  You must use at least 3 different models, and you must ensemble them in some way.  You must also tune your models to improve accuracy.
# 4. (10 points) Accuracy score based on Kaggle submission report (or alternative, see NOTE above).
# 5. (10 points) PEP-8, grammar, spelling, style, etc.
#
# Some additional notes:
#
# 1. If you use another notebook, code, or approaches be sure to reference the original work. (Note that we recommend you study existing Kaggle notebooks before starting your own work.)
# 2. You can help each other but in the end you must submit your own work, both to Kaggle and to Canvas.
#
# Some additional resources:
#
# * "ensemble" your models with a [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
# * a good primer on [feature engineering](https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/)
# * There are a lot of good [notebooks to study](https://www.kaggle.com/competitions/spaceship-titanic/code) (check the number of upvotes to help guide your exploration), but be careful to cite any code that you use, and be careful to not accidentally (or intentionally) cheat.
#
# ## GOOD LUCK!
# (and don't cheat)

# %% [markdown]
# One final note:  Your submission should be a self-contained notebook that is NOT based
# on this one.  Studying the existing Kaggle competition notebooks should
# give you a sense of what makes a "good" notebook.

# %% [markdown]
# # Exploratory Data Analyses

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_palette("pastel")

# %%
titanic_train = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")
titanic_train.head()

# %% [markdown]
# ## General Description

# %%
titanic_train.describe()

# %%
titanic_train.info()

# %%
titanic_train.isnull().sum()

# %% [markdown]
# ## Categorical Features

# %%
titanic_train_cat = titanic_train.select_dtypes(include=["object", "bool"])
titanic_train_cat.drop(["PassengerId", "Name"], axis=1, inplace=True)
titanic_train_cat.head()

# %%
titanic_train_cat["Cabin"] = titanic_train_cat["Cabin"].apply(
    lambda x: x[0] if pd.notnull(x) else None
)
titanic_train_cat["Cabin"].value_counts()

# %%
fig, ax = plt.subplots(2, 3, figsize=(16, 9))

for i, col in enumerate(titanic_train_cat.columns):
    sns.countplot(
        x=col,
        data=titanic_train_cat,
        order=titanic_train_cat[col].value_counts().index,
        ax=ax[i // 3, i % 3],
    )
    ax[i // 3, i % 3].set_title("the number of passengers by {}".format(col))
    ax[i // 3, i % 3].set_ylabel("the number of passengers")
    ax[i // 3, i % 3].set_xlabel("{}".format(col))

plt.tight_layout()

# %%
transported = titanic_train["Transported"].value_counts()
print(
    "Number of passengers transported:",
    round(transported[0] / len(titanic_train) * 100, 2),
    "%",
)
print(
    "Number of passengers not transported:",
    round(transported[1] / len(titanic_train) * 100, 2),
    "%",
)

# %% [markdown]
# ## Numeric Features

# %%
titanic_train_num = titanic_train.select_dtypes(include=["float64"])
titanic_train_num.head()

# %%
titanic_train_num_trans = titanic_train_num.copy()
titanic_train_num_trans["Transported"] = titanic_train["Transported"]

sns.pairplot(data=titanic_train_num_trans, hue="Transported")

# %%
fig, ax = plt.subplots(2, 3, figsize=(16, 9))

for i, col in enumerate(titanic_train_num.columns):
    titanic_train_num[col].plot(kind="hist", ax=ax[i // 3, i % 3])
    ax[i // 3, i % 3].set_title("the distribution of {}".format(col))
    ax[i // 3, i % 3].set_ylabel("the number of passengers")
    ax[i // 3, i % 3].set_xlabel("{}".format(col))

plt.tight_layout()

# %%
plt.figure(figsize=(16, 9))
sns.heatmap(titanic_train_num.corr(), annot=True, cmap="plasma")

# %% [markdown]
# ## My Explanation
# - General Description:
#     - Threr are 14 columns in total, containing id on each passenger and several features
#     - Nearly all columns have null values, except `PassengerId` and "Transported"
# - Categorial Features:
#     - Most of the passengers come from Earth, not in cryoSleep, in cabin F, to TRAPPIST-1e, not VIP, and  50.36% of them are transported finally
#     - The `Cabin` column need to be processed into 3 parts, i.e., `deck , `num`, `side`, which may be helpful to the prediction
#     - The `PassengerId` column follows the form of `gggg_pp`, i.e., where `gggg` indicates a group the passenger is travelling with and `pp` is their number within the group. They need to be seperated since it may matter.
# - Numeric Features:
#     - The pair plot clearly shows the relationship between the numeric features. By using the `Transported` for hue, it indicates the possible combinations which may be helpful in the featuring process
#     - The `Age` are right-skewed
#     - The heatmap indicated `FoodCourt` has the most positive relationship with `Spa` and `VRDeck`, and `VRDeck` has a most negative relationship with `RoomService`

# %% [markdown]
# # Train the Model

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# %% [markdown]
# First define the pipelines, both for numerical and categorical columns.

# %%
num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

cat_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder()),
    ]
)

num_attribs = list(titanic_train_num) + [
    "Cabin_2",
    "Consumption",
    "id_group",
    "vrdeck_vs_food",
    "spa_vs_food",
    "spa_vs_shopping",
    "room_vs_shopping",
    "vrdeck_vs_shopping",
]
cat_attribs = list(titanic_train_cat.drop(["Transported", "Cabin"], axis=1)) + [
    "Cabin_1",
    "Cabin_3",
    "id_no",
]

full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ]
)

# %% [markdown]
# Preprocess train set.
#
# Here we preprocess "Cabin" and explode it, as well as add some additional  features by combining a few numerical features.

# %%
X_train = titanic_train.drop(["PassengerId", "Cabin", "Name", "Transported"], axis=1)
X_train["Cabin_1"] = titanic_train["Cabin"].apply(
    lambda x: x.split("/")[0] if pd.notnull(x) else None
)
X_train["Cabin_2"] = titanic_train["Cabin"].apply(
    lambda x: x.split("/")[1] if pd.notnull(x) else None
)
X_train["Cabin_3"] = titanic_train["Cabin"].apply(
    lambda x: x.split("/")[2] if pd.notnull(x) else None
)
X_train["Consumption"] = (
    X_train["RoomService"]
    + X_train["FoodCourt"]
    + X_train["ShoppingMall"]
    + X_train["Spa"]
)
X_train["id_group"] = titanic_train["PassengerId"].apply(
    lambda x: x.split("_")[0] if pd.notnull(x) else None
)
X_train["id_no"] = titanic_train["PassengerId"].apply(
    lambda x: x.split("_")[1] if pd.notnull(x) else None
)
X_train["vrdeck_vs_food"] = X_train["VRDeck"] / (X_train["FoodCourt"] + 1)
X_train["vrdeck_vs_food"].replace(np.inf, None, inplace=True)
X_train["vrdeck_vs_shopping"] = X_train["VRDeck"] / (X_train["ShoppingMall"] + 1)
X_train["vrdeck_vs_shopping"].replace(np.inf, None, inplace=True)
X_train["spa_vs_food"] = X_train["Spa"] / (X_train["FoodCourt"] + 1)
X_train["spa_vs_food"].replace(np.inf, None, inplace=True)
X_train["spa_vs_shopping"] = X_train["Spa"] / (X_train["ShoppingMall"] + 1)
X_train["spa_vs_shopping"].replace(np.inf, None, inplace=True)
X_train["room_vs_shopping"] = X_train["RoomService"] / (X_train["ShoppingMall"] + 1)
X_train["room_vs_shopping"].replace(np.inf, None, inplace=True)
# X_train.drop('Cabin', axis=1, inplace=True)
X_train.head()

# %%
X_train_formed = full_pipeline.fit_transform(X_train)

# %%
y_train = titanic_train["Transported"]

# %% [markdown]
# Preprocess test set.

# %%
X_test = titanic_test.drop(["PassengerId", "Cabin", "Name"], axis=1)
X_test["Cabin_1"] = titanic_test["Cabin"].apply(
    lambda x: x.split("/")[0] if pd.notnull(x) else None
)
X_test["Cabin_2"] = titanic_test["Cabin"].apply(
    lambda x: x.split("/")[1] if pd.notnull(x) else None
)
X_test["Cabin_3"] = titanic_test["Cabin"].apply(
    lambda x: x.split("/")[2] if pd.notnull(x) else None
)
X_test["Consumption"] = (
    X_test["RoomService"] + X_test["FoodCourt"] + X_test["ShoppingMall"] + X_test["Spa"]
)
X_test["id_group"] = titanic_test["PassengerId"].apply(
    lambda x: x.split("_")[0] if pd.notnull(x) else None
)
X_test["id_no"] = titanic_test["PassengerId"].apply(
    lambda x: x.split("_")[1] if pd.notnull(x) else None
)
X_test["vrdeck_vs_food"] = X_test["VRDeck"] / (X_test["FoodCourt"] + 1)
X_test["vrdeck_vs_food"].replace(np.inf, None, inplace=True)
X_test["vrdeck_vs_shopping"] = X_test["VRDeck"] / (X_test["ShoppingMall"] + 1)
X_test["vrdeck_vs_shopping"].replace(np.inf, None, inplace=True)
X_test["spa_vs_food"] = X_test["Spa"] / (X_test["FoodCourt"] + 1)
X_test["spa_vs_food"].replace(np.inf, None, inplace=True)
X_test["spa_vs_shopping"] = X_test["Spa"] / (X_test["ShoppingMall"] + 1)
X_test["spa_vs_shopping"].replace(np.inf, None, inplace=True)
X_test["room_vs_shopping"] = X_test["RoomService"] / (X_test["ShoppingMall"] + 1)
X_test["room_vs_shopping"].replace(np.inf, None, inplace=True)
# X_test.drop('Cabin', axis=1, inplace=True)
X_test.head()

# %%
X_test_formed = full_pipeline.fit_transform(X_test)

# %% [markdown]
# ## Naive Bayes

# %%
from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB()
model_nb.fit(X_train_formed, y_train)

# %%
y_pred = model_nb.predict(X_train_formed)
y_pred

# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


# %%
def display_scores(scores):
    print(
        "Accuracy: %0.2f%% (+/- %0.2f)" % (scores.mean() * 100, scores.std() * 2 * 100)
    )


# %%
nb_scores = cross_val_score(model_nb, X_train_formed, y_train, cv=10)
display_scores(nb_scores)

# %% [markdown]
# ## Support Vector Machines

# %%
from sklearn import svm

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    "random_state": [42],
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1, 5],
    "gamma": [0.1, 1],
}

# %%
model_svm = svm.SVC(probability=True)
grid_svm = GridSearchCV(model_svm, param_grid, cv=10, n_jobs=-1)
grid_svm.fit(X_train_formed, y_train)

# %%
grid_svm.best_params_

# %%
model_svm = grid_svm.best_estimator_

# %%
svm_scores = cross_val_score(model_svm, X_train_formed, y_train, cv=10)
display_scores(svm_scores)

# %% [markdown]
# ## Decision Trees

# %%
from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier()
model_dt

# %%
param_grid = {
    "random_state": [42],
    "criterion": ["gini", "entropy"],
    "splitter": ["best"],
    "max_depth": [3, 5, 10, 15, 20],
    "min_samples_split": [2, 3, 5, 10, 15, 20, 30, 40, 50, 60],
}

# %%
grid_dt = GridSearchCV(model_dt, param_grid, cv=10, n_jobs=-1)
grid_dt.fit(X_train_formed, y_train)

# %%
grid_dt.best_params_

# %%
model_dt = grid_dt.best_estimator_

# %%
display_scores(cross_val_score(model_dt, X_train_formed, y_train, cv=10))

# %% [markdown]
# ## Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()

# %%
param_grid = {
    "random_state": [42],
    "criterion": ["gini", "entropy"],
    "n_estimators": [10, 15, 20, 25, 30, 35, 40, 50, 90, 95, 100, 105, 110, 120],
    "max_depth": [2, 5, 7, 9, 11, 13, 15],
}

# %%
grid_rf = GridSearchCV(model_rf, param_grid, cv=10, n_jobs=-1)
grid_rf.fit(X_train_formed, y_train)

# %%
grid_rf.best_params_

# %%
model_rf = grid_rf.best_estimator_
model_rf

# %%
display_scores(cross_val_score(model_rf, X_train_formed, y_train, cv=10))

# %% [markdown]
# ## Adaboost

# %%
from sklearn.ensemble import AdaBoostClassifier

model_ada = AdaBoostClassifier(estimator=None)

# %%
param_grid = {
    "random_state": [42],
    "n_estimators": [10, 15, 20, 25, 30, 35, 40],
    "learning_rate": [0.1, 0.5, 1, 1.5, 2],
}

# %%
grid_ada = GridSearchCV(model_ada, param_grid, cv=10, n_jobs=-1)
grid_ada.fit(X_train_formed, y_train)

# %%
grid_ada.best_params_

# %%
model_ada = grid_ada.best_estimator_

# %%
display_scores(cross_val_score(model_ada, X_train_formed, y_train, cv=10))

# %% [markdown]
# ## Voting

# %%
from sklearn.ensemble import VotingClassifier

# %%
voting_clf = VotingClassifier(
    estimators=[
        ("nb", model_nb),
        ("svm", model_svm),
        ("dt", model_dt),
        ("rf", model_rf),
        ("ada", model_ada),
    ],
    voting="soft",
)

# %%
voting_clf.fit(X_train_formed, y_train)

# %%
voting_clf.get_params()

# %%
display_scores(cross_val_score(voting_clf, X_train_formed, y_train, cv=10))

# %%
y_test = voting_clf.predict(X_test_formed)
y_test

# %%
save_file = pd.DataFrame(
    {"PassengerId": titanic_test["PassengerId"], "Transported": y_test}
)
save_file.to_csv("my_submission.csv", index=False)

# %% [markdown]
# # Conclusion

# %% [markdown]
# The result gets a score of **80.196%** on *Kaggle*
