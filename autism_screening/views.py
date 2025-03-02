import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree, export_graphviz # Visualize Tree / Rules
from sklearn.metrics import f1_score,precision_score, recall_score, accuracy_score, confusion_matrix


def index(request):
    return render(request, "index.html")


def predict(request):
    return render(request, "predict.html")


def result(request):
    #with open('model.pkl', 'rb') as f:
    #    model = pickle.load(f)

    df = pd.read_csv('toddler_autism_dataset.csv')

    # Replace special
    df.columns=df.columns.str.replace('-','_')
    df.columns=df.columns.str.replace('/','_')
    df.columns=df.columns.str.replace(' ','_')

    def binary(YN): # To be completed by Student
        if (YN == 'yes' or YN == 'Yes'):
            return 1
        else:
            return 0

    def mapSex(degree): # To be completed by Student
        if (degree == 'f'):
            return 1
        else:
            return 0

    def mapEthnicity(ethnic): # To be completed by Student
        if ethnic == 'middle eastern':
            return 1
        if ethnic == 'White European':
            return 2
        if ethnic == 'Hispanic':
            return 3
        if ethnic == 'black':
            return 4
        if ethnic == 'asian':
            return 5
        if ethnic == 'south asian':
            return 6
        if ethnic == 'Native Indian':
            return 7
        if ethnic == 'Others':
            return 8
        if ethnic == 'Latino':
            return 9
        if ethnic == 'mixed':
            return 10
        if ethnic == 'Pacifica':
            return 11
        else:
            return 0

    def mapAssessor(assessor): # To be completed by Student
        if(assessor == 'Health Care Professional'):
            return 1
        else:
            return 0


    Sex = mapSex(df['Sex'].any())
    Ethnicity = mapEthnicity(df['Ethnicity'].any())
    Jaundice = binary(df['Jaundice'].any())
    Family_mem_with_ASD = binary(df['Family_mem_with_ASD'].any())


    Class_ASD_Traits_ = pd.Categorical(pd.Categorical(df['Class_ASD_Traits_']).codes)

    df['Sex'] = Sex
    df['Ethnicity'] = Ethnicity
    df['Jaundice'] = Jaundice
    df['Family_mem_with_ASD'] = Family_mem_with_ASD
    df['Class_ASD_Traits_'] = Class_ASD_Traits_

    # Check the data and think why we drop these variables?

    X = df.drop(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10','Case_No', 'Who_completed_the_test', 'Qchat_10_Score', 'Class_ASD_Traits_'], axis=1)

    Y_classification = df.Class_ASD_Traits_

    X_train, X_test, y_train, y_test = train_test_split(X, Y_classification, test_size=1 / 3, random_state=1, stratify=Y_classification)
    dt = DecisionTreeClassifier(criterion='gini',random_state=0, max_depth=8)
    dt.fit(X_train, y_train)

    #pd.Categorical(pd.Categorical(request.GET['n1']).codes)

    val1 = float(request.GET['n1'])
    val2 = mapSex(request.GET['n2'])
    val3 = mapEthnicity(request.GET['n3'])
    val4 = binary(request.GET['n4'])
    val5 = binary(request.GET['n5'])
    #val5 = float(request.GET['n5'])

    pred = dt.predict([[val1, val2, val3, val4, val5]])
    #pred = dt.predict([[22,1,8,1,1]])

    #result1 = val1 + val2 + val3 + val4 + val5
    if pred == [1]:
       result1 = "Future Genius!"
    else:
       result1 = "Well.. No!"

    return render(request, "predict.html",
                  {"result2": result1})
