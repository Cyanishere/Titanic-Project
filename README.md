# Titanic Survival Prediction

## Project Overview

This project aims to predict the survival of passengers on the Titanic using various machine learning algorithms. 
The dataset includes information about the passengers, such as age, gender, class, and fare, which are used to train models to predict survival outcomes.

## Dataset

The dataset used for this project is the Titanic dataset, which can be found from kaggle (https://www.kaggle.com/c/titanic/data). 
The dataset includes the following features:

- **PassengerId**: Unique identifier for each passenger
- **Pclass**: Ticket class (1st, 2nd, or 3rd)
- **Name**: Name of the passenger
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings or spouses aboard the Titanic
- **Parch**: Number of parents or children aboard the Titanic
- **Ticket**: Ticket number
- **Fare**: Fare paid by the passenger
- **Cabin**: Cabin number (not used in this analysis)
- **Embarked**: Port of embarkation (not used in this analysis)
- **Survived**: Survival status (0 = No, 1 = Yes)

## Project Structure

First, prior data analysis and processing is performed with the following principles:
- Correlation: Understand the correlation between each features and target.
- Completing: Fill or Drop the missing values(Age, Embarked...).
- Correcting: Drop un-nesscessary columns(only select the useful features for model training).
- Creating: Combine two features or add a new features, e.g. combine SibSp and Parch to get FamilySize.

Secondly, split the datasets into training and testing datasets (0.8 training and 0.2 testing)

Thirdly, trian the Machine Learning Models.
In this project, the following machine learning models were implemented:

Logistic Regression
Random Forest Classifier
K-Neighbors Classifier
Support Vector Machine
Gaussian Naive Bayes
Decision Tree Classifier
Stochastic Gradient Descent
Perceptron

## Results
The best performing model was the Random Forest Classifier, achieving a score of 0.821 on the test set.


## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

You can install the required packages using the following command after
[cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

```bash
pip install -r requirements.txt
```

## Directory Tree
Titanic/
│
├── Titanic.ipynb # Jupyter notebook containing the analysis and model training
├── titanic.csv # Titanic dataset
├── requirements.txt # List of required Python packages
└── README.md # Project documentation


## Conclusion
This project demonstrates the application of various machine learning techniques for predicting survival on the Titanic. 
It provides insights into the impact of different features on survival rates.

## Author
Jasper Lung
linkedin.com/in/jasper-lung-95aa29242




