import numpy as np
import matplotlib.pyplot as plt
import argparse

from modeling import (
    fit_linear_model, eval_linear, eval_polynomial, 
    compute_polynomial_basis_funcs, normalize, 
    fit_linear_model_with_ridge_regression)
from visualization import plot_points, plot_function

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_npz_path", type=str, required=False,default="./crop_data.npz",
        help="Path to the .npz file containing exercise data")
    parser.add_argument(
        "--ridge_lambda", type=float, default=0.7, required=False,
        help="Lambda hyperparameter for ridge regression")
    parser.add_argument(
        "--num_training_samples", type=int, default=300, required=False,
        help="The number of training samples to use (<= 300)")
    parser.add_argument(
        "--max_degree", type=int, default=7, required=False,
        help="The maximum degree to be used in the polynomial basis functions")

    args = parser.parse_args()
    data_path = args.data_npz_path
    ridge_lambda = args.ridge_lambda
    max_degree = args.max_degree
    num_training_samples = args.num_training_samples

    data = np.load(data_path)
    x_train, y_train, x_val, y_val = (
        data["x_train"], data["y_train"], data["x_val"], data["y_val"])
    #select a subset of the training data
    x_train = x_train[:num_training_samples]
    y_train = y_train[:num_training_samples]
    #normalize for improved numerical stability
    x_train, m_x_train, std_x_train = normalize(x_train)
    y_train, m_y_train, std_y_train = normalize(y_train)
    #important: validation data always gets regularized with training data
    #statistics
    x_val, _, _ = normalize(x_val, m_x_train, std_x_train)
    y_val, _, _ = normalize(y_val, m_y_train, std_y_train)

    w_linear, b_linear = fit_linear_model(x_train, y_train)
    w_linear_reg, b_linear_reg = fit_linear_model_with_ridge_regression(
        x_train, y_train, ridge_lambda)

    fig, ax = plt.subplots(1, 2)
    plot_min = np.min(np.hstack((x_train, x_val))) - 0.05
    plot_max = np.max(np.hstack((x_train, x_val))) + 0.05
    for a in ax:
        plot_points(np.stack((x_train, y_train), axis=1), a, 'x', "red")
        plot_points(np.stack((x_val, y_val), axis=1), a, 'o', "blue")

    handles = [plot_function(lambda x : eval_linear(x, w_linear, b_linear), 
                             plot_min, plot_max, ax[0])]
    handles_reg = [plot_function(
        lambda x : eval_linear(x, w_linear_reg, b_linear_reg), plot_min, 
        plot_max, ax[1])]

    mse_train= [
        np.mean(np.square(eval_linear(x_train, w_linear, b_linear) - y_train))]
    mse_val = [
        np.mean(np.square(eval_linear(x_val, w_linear, b_linear) - y_val))]
    mse_train_reg = [
        np.mean(np.square(eval_linear(x_train, w_linear, b_linear) - y_train))]
    mse_val_reg = [np.mean(np.square(eval_linear(x_val, w_linear_reg, 
                                                 b_linear_reg) - y_val))]
    legend = ["Linear, MSE train: {0:.3f} MSE val: {1:.3f}"
              .format(mse_train[-1], mse_val[-1])]
    legend_reg = ["Linear, MSE train: {0:.3f} MSE val: {1:.3f}".format(
        mse_train_reg[-1], mse_val_reg[-1])]

    for m in range(2, max_degree + 1):
        w, b = fit_linear_model(compute_polynomial_basis_funcs(
            x_train, m), y_train)
        w_reg, b_reg = fit_linear_model_with_ridge_regression(
            compute_polynomial_basis_funcs(x_train, m), y_train, ridge_lambda)
        
        mse_train.append(np.mean(
            np.square(eval_polynomial(x_train, m, w, b) - y_train)))
        mse_val.append(np.mean(
            np.square(eval_polynomial(x_val, m, w, b) - y_val)))
        mse_train_reg.append(np.mean(
            np.square(eval_polynomial(x_train, m, w_reg, b_reg) - y_train)))
        mse_val_reg.append(np.mean(
            np.square(eval_polynomial(x_val, m, w_reg, b_reg) - y_val)))        
        
        text = "Polynomial, deg. {0:.3f}, MSE train: {1:.3f} MSE val: {2:.3f}"
        legend.append(text.format(m, mse_train[-1], mse_val[-1]))
        legend_reg.append(text.format(m, mse_train_reg[-1], mse_val_reg[-1]))

        eval_func = lambda x : eval_polynomial(x, m, w, b)
        handles.append(plot_function(eval_func, plot_min, plot_max, ax[0]))
        eval_func = lambda x : eval_polynomial(x, m, w_reg, b_reg)
        handles_reg.append(plot_function(eval_func, plot_min, plot_max, ax[1]))
        
    ax[0].legend(labels=legend, handles=handles)
    ax[0].set_title("Regression without regularization")
    ax[0].set_xlabel("Independent variable")
    ax[0].set_ylabel("Dependent variable")
    ax[1].legend(labels=legend_reg, handles=handles_reg)
    ax[1].set_title("Regression with l-2 regularization")
    ax[1].set_xlabel("Independent variable")
    ax[1].set_ylabel("Dependent variable")    
    plt.show()