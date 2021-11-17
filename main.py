import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
# 1a


def getNullValues():
    df = pd.read_csv('train.csv')
    print(df.isnull().sum())
    print(df.shape)

# 1b


def getTenRows():
    df = pd.read_csv('train.csv')
    df = df[df.isnull().T.any().T]
    df.insert(df.shape[1], 'Data-Entries',
              [12 - df.iloc[x].isnull().sum() for x in range(104)])
    print(df.head(9))

# 1c


def getCommon(df, column):
    return df[column].mode()


def fillMissing():
    df = pd.read_csv('train.csv')
    # 12 , 14 ,17, 130, 234, 466
    print(df.iloc[12, :], "\n", df.iloc[14, :], "\n",
          df.iloc[17, :], "\n", df.iloc[130, :], "\n", df.iloc[234, :], "\n", df.iloc[466, :])
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(df)
    df = pd.DataFrame(imputer.transform(df))
    print(df)
    # print(df.iloc[12, :], "\n", df.iloc[14, :], "\n",
    #       df.iloc[17, :], "\n", df.iloc[130, :], "\n", df.iloc[234, :], "\n", df.iloc[466, :])
    return df

# 1d


def replaceCategorical():
    df = fillMissing()
    df[1].replace(to_replace=['Male', 'Female'],
                  value=[1, 2], inplace=True)
    df[6].replace(
        to_replace=['< 1 Year', '1-2 Year', '> 2 Years'], value=[1, 2, 3], inplace=True)
    df[7].replace(
        to_replace=['Yes', 'No'], value=[1, 0], inplace=True)
    df = df.rename(columns={0: 'id', 1: 'Gender', 2: 'Age', 3: 'Driving_license', 4: 'Region_Code', 5: 'Previously_Insured',
                            6: 'Vehicle_Age', 7: 'Vehicle_Damage', 8: 'Annual_Premium', 9: 'Policy_Sales_Channel', 10: 'Vintage', 11: 'Response'})
    print(df.head(10))
    return df

# 2


def plots():
    # Age Range of Drivers with Annual Premium
    df = pd.DataFrame(replaceCategorical())
    x = df["Age"]
    y = df["Annual_Premium"]
    # plot 1
    plt.scatter(x, y, s=2, c="red")
# 3


def SVM():
    df = replaceCategorical()
    # Unwanted Columns - id, policy_sales_channel, response
    features = df[['Gender', 'Age', 'Driving_license',
                   'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Annual_Premium', 'Vintage']]

    # Independent Variables

    x = np.asarray(features)

    # Dependent Variables

    y = np.asarray(df['Vehicle_Damage'])

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=3161)

    classifierModel = svm.SVC(kernel='linear', gamma='auto', C=2)

    classifierModel.fit(X_train, y_train)

    y_predict = classifierModel.predict(X_test)

    # Shows the accuracy based on the precision and f-score
    print(classification_report(y_test, y_predict))


# getNullValues()
# getTenRows()
# fillMissing()
# replaceCategorical()
# SVM()
plots()
