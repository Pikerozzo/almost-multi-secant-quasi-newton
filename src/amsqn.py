import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from problems import *
from algorithms import *


def main():
    print("Running example...\n")

    # Load the local dataset (source: https://www.kaggle.com/mirichoi0218/insurance)
    insurance_df = pd.read_csv('..\\data\\insurance.csv')

    # Convert the 'smoker' column to a binary column
    insurance_df['smokes'] = (insurance_df['smoker']=='yes').astype(int)

    # Split the dataset into features and target variable
    X, y = insurance_df[['bmi', 'charges']].values, insurance_df['smokes'].values

    # Split the dataset into training and testing sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1492)

    # Scale the features
    sc = StandardScaler()
    sc = sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    # Create the problem instance
    problem = L2LogisticReg(X_train, y_train)

    # Create the algorithm instance and train the model
    alg = AMSQN(problem=problem, mode=AMSQN_Mode.BOTH)
    alg.fit(learning_rate=0, max_iterations=1000)

    # Evaluate the results on the test set
    alg.problem.evaluate(X=X_test, y_true=y_test)

    # Plot the cost function
    plt.plot(alg.problem.costs)
    plt.loglog()
    plt.title('COST')
    plt.show()

    # Plot the QN violation
    plt.plot(alg.qn_violation)
    plt.loglog()
    plt.title('QN VIOLATION')
    plt.show()
           
           
if __name__ == "__main__":
    main()    