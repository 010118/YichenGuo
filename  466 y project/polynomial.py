import numpy as np
import utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():

    # Hyperparameters
    degree = 2  
    
    # training
    X_train, X_val, X_test, y_train, y_val, y_test = utils.getData("red")

    # Polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    X_test_poly = poly.transform(X_test)

    # Linear regression with polynomial features
    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, y_train)

    # Training evaluation
    pred_train = poly_reg.predict(X_train_poly)
    pred_train = np.rint(pred_train)
    y_train = np.rint(np.array(y_train))
    print("Training:")
    print("Mean squared error: " + str(mean_squared_error(y_train, pred_train)))
    print("Mean absolute error: " + str(mean_absolute_error(y_train, pred_train)))
    print("Accuracy: " + str(utils.accuracy(pred_train, y_train)))

    # Validation evaluation
    pred_val = poly_reg.predict(X_val_poly)
    pred_val = np.rint(pred_val)
    y_val_rounded = np.rint(np.array(y_val))
    print("\nValidation:")
    print("Mean squared error: " + str(mean_squared_error(y_val_rounded, pred_val)))
    print("Mean absolute error: " + str(mean_absolute_error(y_val_rounded, pred_val)))
    print("Accuracy: " + str(utils.accuracy(pred_val, y_val)))

    # Testing evaluation
    pred_test = poly_reg.predict(X_test_poly)
    pred_test = np.rint(pred_test)
    y_test_rounded = np.rint(np.array(y_test))
    print("\nTesting:")
    print("Mean squared error: " + str(mean_squared_error(y_test_rounded, pred_test)))
    print("Mean absolute error: " + str(mean_absolute_error(y_test_rounded, pred_test)))
    print("Accuracy: " + str(utils.accuracy(pred_test, y_test)))
    print("Close Accuracy (+-1 score): " + str(utils.close_accuracy(pred_test, y_test)))

if __name__ == "__main__":
    main()