# -*- coding: utf-8 -*-

import pandas as pd
import csv as csv

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier


class Titanic(object):
    def __init__(self, train_file, test_file):
        self.train = pd.read_csv(train_file)
        self.test = pd.read_csv(test_file)

    def data_clean(self):
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

        # drop unnecessary columns
        self.train = self.train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        self.test = self.test.drop(['Name', 'Ticket', 'Cabin'], axis=1)


    def feature_engineering(self):
        """
        feature engineering
        """

        def get_person(age):
            person = ''

            if age <= 16:
                person = 'Child'
            elif age <= 32:
                person = 'Young'
            elif age <= 48:
                person = 'Middle'
            elif age <= 64:
                person = 'Mature'
            else:
                person = 'Old'

            return person

        def get_fare_level(fare):
            level = ''
            if fare <= 7.91:
                level = 'lower'
            elif fare <= 14.454:
                level = 'low'
            elif fare <= 31:
                level = 'middle'
            else:
                level = 'high'

            return level

        # feature: Pclass
        pclass_dummies_train = pd.get_dummies(self.train['Pclass'], prefix='Pclass')
        pclass_dummies_test = pd.get_dummies(self.test['Pclass'], prefix='Pclass')

        # feature: Sex
        self.train['Sex'] = self.train['Sex'].map({'female': 0, 'male': 1}).astype(int)
        self.test['Sex'] = self.test['Sex'].map({'female': 0, 'male': 1}).astype(int)

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

        # feature: fare
        self.train['Fare'] = self.train['Fare'].apply(get_fare_level)
        self.test['Fare'] = self.test['Fare'].apply(get_fare_level)
        fare_dummies_train = pd.get_dummies(self.train['Fare'], prefix='Fare')
        fare_dummies_test = pd.get_dummies(self.test['Fare'], prefix='Fare')


        # concat
        self.X_train = pd.concat([self.train[['Sex', 'IsAlone']], pclass_dummies_train,
                                  person_dummies_train, fare_dummies_train, embarked_dummies_train],
                                 axis=1)
        self.Y_train = self.train['Survived']
        self.X_test = pd.concat([self.test[['Sex', 'IsAlone']], pclass_dummies_test,
                                 person_dummies_test, fare_dummies_test,
                                 embarked_dummies_test], axis=1)


        # feature selection


    def train(self):
        logreg = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        logreg.fit(self.X_train, self.Y_train)
        self.Y_pred = logreg.predict(self.X_test)
        logreg.score(self.X_train, self.Y_train)

        coeff_df = pd.DataFrame({
            'Features': self.X_train.columns,
            'Coefficient': pd.Series(logreg.coef_[0])
        })


    def predict(self):
        result = pd.DataFrame({
            'PassengerId': self.test['PassengerId'],
            'Survived': self.Y_pred
        })

    def evaluate(self):
        pass


    def ensemble(self):






# https://www.kaggle.com/arthurtok/titanic/introduction-to-ensembling-stacking-in-python
# http://localhost:6768/notebooks/A-Journey-through-Titanic.ipynb

# http://blog.csdn.net/han_xiaoyang/article/details/49797143
# http://mars.run/2015/11/Machine%20learning%20kaggle%20titanic-0.8/
