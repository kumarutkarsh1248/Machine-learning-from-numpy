from methods import Linear_Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/home/kumarutkarsh/Desktop/Machine learning from numpy/example/linear_regression/salary_dataset.csv")
data = data.to_numpy()

# min max normalization
X_train = data[:, 1:2]
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
y_train = data[:, 2:]
y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())

plt.scatter(X_train, y_train)

print(X_train)
def main():
    linear_regression = Linear_Regression(X_train, y_train)
    linear_regression.fit()

    print("prediction")
    print("weight ", linear_regression.weights)
    print("bias ", linear_regression.bias)
    print(linear_regression.predict(X_train))

    plt.scatter(X_train, linear_regression.predict(X_train))
    plt.show()


if __name__ == "__main__":
    main()

