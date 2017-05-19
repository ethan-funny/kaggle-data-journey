# -*- coding: utf-8 -*-

import pandas as pd
import csv as csv

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score


class Titanic(object):
    def __init__(self, train_file, test_file):
        self.train = pd.read_csv(train_file)
        self.test = pd.read_csv(test_file)

    def data_preprocess(self):
        """
        data preprocess
        """
        def generate_random_age(df):
            ave_age = df['Age'].mean()
            std_age = df['Age'].std()
            count_null_age = df['Age'].isnull().sum()

            random_age = np.random.randint(ave_age - std_age, ave_age + std_age, size=count_null_age)

            return random_age

        # only for train, fill the missing "Embarked" values with the most occured value
        self.train["Embarked"].fillna(self.train["Embarked"].mode()[0], inplace=True)

        # only for test, fill one missing "Fare" value
        self.test["Fare"].fillna(self.test["Fare"].median(), inplace=True)

        # for train and test, fill missing "Age" values
        self.train.loc[self.train.Age.isnull(), "Age"] = generate_random_age(self.train)
        self.test.loc[self.test.Age.isnull(), "Age"] = generate_random_age(self.test)

        self.train["Age"] = self.train["Age"].astype(int)
        self.test["Age"] = self.test["Age"].astype(int)

    def feature_engineering(self):
        """
        feature engineering
        """

        def get_person(age):
            person = ''

            if age < 18:
                person = 'Child'
            elif age <= 50:
                person = 'Mature'
            else:
                person = 'Old'

            return person

        def get_fare_level(fare):
            level = ''
            if fare <= 14.454:
                level = 'low'
            elif fare <= 25:
                level = 'middle'
            else:
                level = 'high'

            return level


        # feature: Pclass
        pclass_dummies_train = pd.get_dummies(self.train['Pclass'], prefix='Pclass')
        pclass_dummies_test = pd.get_dummies(self.test['Pclass'], prefix='Pclass')

        # feature: Sex
        # self.train['Sex'] = self.train['Sex'].map({'female': 0, 'male': 1}).astype(int)
        # self.test['Sex'] = self.test['Sex'].map({'female': 0, 'male': 1}).astype(int)

        gender_dummies_train = pd.get_dummies(self.train['Sex'], prefix='Gender')
        gender_dummies_test = pd.get_dummies(self.test['Sex'], prefix='Gender')

        # feature: Embarked
        embarked_dummies_train = pd.get_dummies(self.train['Embarked'], prefix='Embarked')
        embarked_dummies_test = pd.get_dummies(self.test['Embarked'], prefix='Embarked')

        # create new feature: Person
        self.train['Person'] = self.train['Age'].apply(get_person)
        self.test['Person'] = self.test['Age'].apply(get_person)
        person_dummies_train = pd.get_dummies(self.train['Person'], prefix='Person')
        person_dummies_test = pd.get_dummies(self.test['Person'], prefix='Person')

        # create new feature: IsAlone
        family_size = self.train['SibSp'] + self.train['Parch']
        self.train['IsAlone'] = family_size.apply(lambda x: 1 if x == 0 else 0)
        self.test['IsAlone'] = family_size.apply(lambda x: 1 if x == 0 else 0)

        isalone_dummies_train = pd.get_dummies(self.train['IsAlone'], prefix='IsAlone')
        isalone_dummies_test = pd.get_dummies(self.test['IsAlone'], prefix='IsAlone')

        # feature: fare
        self.train['Fare'] = self.train['Fare'].apply(get_fare_level)
        self.test['Fare'] = self.test['Fare'].apply(get_fare_level)
        fare_dummies_train = pd.get_dummies(self.train['Fare'], prefix='Fare')
        fare_dummies_test = pd.get_dummies(self.test['Fare'], prefix='Fare')


        # concat
        self.X_train = pd.concat([gender_dummies_train, pclass_dummies_train,
                                  person_dummies_train, embarked_dummies_train],
                                 axis=1)
        self.Y_train = self.train['Survived']
        self.X_test = pd.concat([gender_dummies_test, pclass_dummies_test,
                                 person_dummies_test,
                                 embarked_dummies_test], axis=1)

        # split_train, split_validation = train_test_split(train, test_size = 0.3, random_state = 0)

    def split_data(self, train_data):
        # split data to train and test
        split_train, split_cv = train_test_split(train_data, test_size=0.2,
                                                 random_state=0)

        x_train = split_train.as_matrix()[:, 1:]
        y_train = split_train.as_matrix()[:, 0]
        x_test = split_cv.as_matrix()[:, 1:]
        y_test = split_cv.as_matrix()[:, 0]


    @staticmethod
    def lr(x_train, y_train, x_test, y_test, columns):
        # train and predict
        logreg = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        logreg.fit(x_train, y_train)

        y_pred = logreg.predict(x_test)

        # evaluate
        coeff = logreg.coef_[0]

        coeff_df = pd.DataFrame({
            'Coefficient': pd.Series(coeff),
            'Features': columns
        })

        result = Titanic.evaluate(y_test, y_pred)

        result.update({'model': logreg, 'y_pred': y_pred, 'coeff': coeff_df})

        return result


    @staticmethod
    def rf(x_train, y_train, x_test, y_test):
        # train and predict
        random_forest = RandomForestClassifier(n_estimators=1000)
        random_forest.fit(x_train, y_train)

        y_pred = random_forest.predict(x_test)

        # evaluate
        result = Titanic.evaluate(y_test, y_pred)

        result.update({'model': random_forest, 'y_pred': y_pred})

        return result

    @staticmethod
    def gbdt(x_train, y_train, x_test, y_test):

        # train and predict
        GBDT = GradientBoostingClassifier(n_estimators=1000)
        GBDT.fit(x_train, y_train)

        y_pred = GBDT.predict(x_test)

        # evaluate
        result = Titanic.evaluate(y_test, y_pred)

        result.update({'model': GBDT, 'y_pred': y_pred})

        return result

    @staticmethod
    def evaluate(y_true, y_pred):
        cmatrix = confusion_matrix(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        score_df = pd.DataFrame(
            [[auc, accuracy, recall]],
            columns=['auc', 'accuracy', 'recall']
        )
        cmatrix_df = pd.DataFrame(
            cmatrix,
            columns=['0(predict)', '1(predict)'],
            index=['0(actual)', '1(actual)']
        )

        print('====score====')
        display(score_df)

        print('====confusion matrix====')
        display(cmatrix_df)

        return dict(score=score_df, cmatrix=cmatrix_df)

    def ensemble(self, y_lr, y_rf, y_gbdt, y_true):

        res = pd.DataFrame({
            'lr': y_lr,
            'rf': y_rf,
            'gbdt': y_gbdt,
        })

        res.insert(res.shape[1], 'predict', res.mode(axis=1))

        res.insert(res.shape[1], 'actual', y_true)



    @staticmethod
    def generate_result(model, test_data, passenger_id):
        y_pred = model.predict(test_data)

        result = pd.DataFrame({
            'PassengerId': passenger_id,
            'Survived': y_pred
        })

        result.to_csv('titanic_solution.csv', index=False)


    def tocsv(self):
        result = pd.DataFrame({
            'PassengerId': self.test['PassengerId'],
            'Survived': self.Y_pred
        })
        result.to_csv('titanic_solution.csv', index=False)



if __name__ == '__main__':
    titanic = Titanic('train.csv', 'test.csv')
    print('data clean...')
    titanic.data_clean()

    print('feature engineering...')
    titanic.feature_engineering()

    print('training...')
    titanic.model_train()

    print('predict...')
    titanic.predict()

    print('to csv...')
    titanic.tocsv()

    print('evaluating...')
    titanic.evaluate()



# 特征交叉
# 其他特征

# https://www.kaggle.com/arthurtok/titanic/introduction-to-ensembling-stacking-in-python
# http://localhost:6768/notebooks/A-Journey-through-Titanic.ipynb

# http://blog.csdn.net/han_xiaoyang/article/details/49797143
# http://mars.run/2015/11/Machine%20learning%20kaggle%20titanic-0.8/

# # model.predict_proba(split_cv.as_matrix()[:, 1:])