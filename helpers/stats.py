import os
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import norm
import numpy as np

def plot_error_distributions(best_sensor_rmse, best_sensor_mae, best_sensor_accuracy, best_sensor_r2, best_sensor_variance, output_path):
    
    save_path = os.path.join(output_path, "stats")

    ### normal distribution of the min rmse values for all sensors nodes ###
    # sort the values for the distributions
    best_sensor_rmse = sorted(best_sensor_rmse)

    print("Generating RMSE stats...")

    # fit the curve normal distribution #cdf alt
    fit_data = stats.norm.cdf(best_sensor_rmse, np.mean(
        best_sensor_rmse), np.std(best_sensor_rmse))  # this is a fitting indeed

    # title
    plt.title("Normal Distribution of the Min RMSE Values for All Sensors", fontweight="bold",
              color="#333333", pad=10, fontname='Times New Roman', fontdict={'fontsize': 12})
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)

    # plot the distribution σ µ
    plt.plot(best_sensor_rmse, fit_data, linewidth=1, marker=".", markersize=2, color="#ffd700", label="RMSE, σ = " +
             str(round(np.std(best_sensor_rmse), 2)) + ", µ = " + str(round(np.mean(best_sensor_rmse), 2)),)
    plt.legend(loc='best', fontsize=8)

    # plot the mean
    plt.axvline(x=np.mean(best_sensor_rmse), linewidth=1,
                color='#ffd700', ls='--', label='axvline - full height')

    # axis titles
    plt.xlabel('Test Root Mean Square Error (RMSE)', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.ylabel('Frequency in %', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)

    # plt.hist(h)      #use this to draw histogram of your data

    # save the distribution
    plt.savefig(save_path+'/rmse_distribution.jpg', dpi=500)

    # show the distribution
    # plt.show()

    # close the distribution
    plt.close()

    ### normal distribution of the min mae values for all sensors nodes ###
    # sort the values for the distributions
    best_sensor_mae = sorted(best_sensor_mae)

    print("Generating MAE stats...")

    # fit the curve normal distribution
    fit_data = stats.norm.cdf(best_sensor_mae, np.mean(
        best_sensor_mae), np.std(best_sensor_mae))  # this is a fitting indeed

    # title
    plt.title("Normal Distribution of the Min MAE Values for All Sensors", fontweight="bold",
              color="#333333", pad=10, fontname='Times New Roman', fontdict={'fontsize': 12})
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)

    # plot the distribution σ µ
    plt.plot(best_sensor_mae, fit_data, linewidth=1, marker=".", markersize=2, color="#ffd700", label="MAE, σ = " +
             str(round(np.std(best_sensor_mae), 2)) + ", µ = " + str(round(np.mean(best_sensor_mae), 2)),)
    plt.legend(loc='best', fontsize=8)

    # plot the mean
    plt.axvline(x=np.mean(best_sensor_mae), linewidth=1,
                color='#ffd700', ls='--', label='axvline - full height')

    # axis titles
    plt.xlabel('Test Mean Absolute Error (MAE)', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.ylabel('Frequency in %', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)

    # plt.hist(h)      #use this to draw histogram of your data

    # save the distribution
    plt.savefig(save_path+'/mae_distribution.jpg', dpi=500)

    # show the distribution
    # plt.show()

    # close the distribution
    plt.close()

    ### normal distribution of the max acc values for all sensors nodes ###
    # sort the values for the distributions
    best_sensor_accuracy = sorted(best_sensor_accuracy)

    print("Generating Acc stats...")

    # fit the curve normal distribution
    fit_data = stats.norm.cdf(best_sensor_accuracy, np.mean(
        best_sensor_accuracy), np.std(best_sensor_accuracy))  # this is a fitting indeed

    # title
    plt.title("Normal Distribution of the Max Accuracy Values for All Sensors", fontweight="bold",
              color="#333333", pad=10, fontname='Times New Roman', fontdict={'fontsize': 12})
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)

    # plot the distribution σ µ
    plt.plot(best_sensor_accuracy, fit_data, linewidth=1, marker=".", markersize=2, color="#ffd700", label="Acc, σ = " +
             str(round(np.std(best_sensor_accuracy), 2)) + ", µ = " + str(round(np.mean(best_sensor_accuracy), 2)),)
    plt.legend(loc='best', fontsize=8)

    # plot the mean
    plt.axvline(x=np.mean(best_sensor_accuracy), linewidth=1,
                color='#ffd700', ls='--', label='axvline - full height')

    # axis titles
    plt.xlabel('Test (Accuracy)', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.ylabel('Frequency in %', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.axis([-1, 1, 0, None])
    # plt.hist(h)      #use this to draw histogram of your data

    # save the distribution
    plt.savefig(save_path+'/acc_distribution.jpg', dpi=500)

    # show the distribution
    # plt.show()

    # close the distribution
    plt.close()

    ### normal distribution of the max r2 values for all sensors nodes ###
    # sort the values for the distributions
    best_sensor_r2 = sorted(best_sensor_r2)

    print("Generating R2 stats...")

    # fit the curve normal distribution
    fit_data = stats.norm.cdf(best_sensor_r2, np.mean(
        best_sensor_r2), np.std(best_sensor_r2))  # this is a fitting indeed

    # title
    plt.title("Normal Distribution of the Max R² Values for All Sensors", fontweight="bold",
              color="#333333", pad=10, fontname='Times New Roman', fontdict={'fontsize': 12})
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)

    # plot the distribution σ µ
    plt.plot(best_sensor_r2, fit_data, marker=".", linewidth=1, markersize=2, color="#ffd700", label="R², σ = " +
             str(round(np.std(best_sensor_r2), 2)) + ", µ = " + str(round(np.mean(best_sensor_r2), 2)),)
    plt.legend(loc='best', fontsize=8)
    plt.axis([-1, 1, 0, None])

    # plot the mean
    plt.axvline(x=np.mean(best_sensor_r2), linewidth=1,
                color='#ffd700', ls='--', label='axvline - full height')

    # axis titles
    plt.xlabel('Test Coefficient of Determination (R²)', fontweight="bold",
               color="#333333", fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.ylabel('Frequency in %', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)

    # plt.hist(h)      #use this to draw histogram of your data

    # save the distribution
    plt.savefig(save_path+'/r2_distribution.jpg', dpi=500)

    # show the distribution
    # plt.show()

    # close the distribution
    plt.close()

    ### normal distribution of the max var values for all sensors nodes ###
    # sort the values for the distributions
    best_sensor_variance = sorted(best_sensor_variance)

    print("Generating Variance stats...")

    # fit the curve normal distribution
    fit_data = stats.norm.cdf(best_sensor_variance, np.mean(
        best_sensor_variance), np.std(best_sensor_variance))  # this is a fitting indeed

    # title
    plt.title("Normal Distribution of the Max VAR Values for All Sensors", fontweight="bold",
              color="#333333", pad=10, fontname='Times New Roman', fontdict={'fontsize': 12})
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)

    # plot the distribution σ µ
    plt.plot(best_sensor_variance, fit_data, marker=".", linewidth=1, markersize=2, color="#ffd700", label="VAR, σ = " +
             str(round(np.std(best_sensor_variance), 2)) + ", µ = " + str(round(np.mean(best_sensor_variance), 2)),)
    plt.legend(loc='best', fontsize=8)
    plt.axis([-1, 1, 0, None])

    # plot the mean
    plt.axvline(x=np.mean(best_sensor_variance), linewidth=1,
                color='#ffd700', ls='--', label='axvline - full height')

    # axis titles
    plt.xlabel('Test Variance (VAR)', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.ylabel('Frequency in %', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)

    # plt.hist(h)      #use this to draw histogram of your data

    # save the distribution
    plt.savefig(save_path+'/var_distribution.jpg', dpi=500)

    # show the distribution
    # plt.show()

    # close the distribution
    plt.close()


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def normal_dist(x, mean, sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density


def plot_additional_errors(test_rmse, train_rmse, train_loss, alpha1, save_path):
    # plot additional errors for the model
    # train_rmse & test_rmse
    fig1 = plt.figure(figsize=(6, 4))
    plt.title(str("Training vs Test Root Mean Square Error (RMSE)"), fontweight="bold",
              color="#333333", pad=10, fontname='Times New Roman', fontdict={'fontsize': 12})
    # x y labels
    plt.xlabel('Epochs', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.ylabel('RMSE', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.plot(train_rmse, label="train_rmse", color="#ffd700")
    plt.plot(test_rmse, label="test_rmse", color="#333333")
    plt.legend(loc='best', fontsize=8)
    # save the figure
    plt.savefig(save_path+'/rmse.jpg', dpi=500)
    # plt.show()
    # close the figure
    plt.close()

    # train_loss
    fig1 = plt.figure(figsize=(6, 4))
    plt.title(str("Training Loss"), fontweight="bold", color="#333333",
              pad=10, fontname='Times New Roman', fontdict={'fontsize': 10})
    # x y labels
    plt.xlabel('Epochs', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.ylabel('Loss', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.plot(train_loss, label='train_loss', color="#333333")
    plt.yscale('log')
    plt.legend(loc='best', fontsize=8)
    plt.savefig(save_path+'/train_loss.jpg', dpi=500)
    # save the figure
    # plt.show()
    # close the figure
    plt.close()

    # train_rmse
    fig1 = plt.figure(figsize=(6, 4))
    plt.title(str("Training Root Mean Square Error (RMSE)"), fontweight="bold",
              color="#333333", pad=10, fontname='Times New Roman', fontdict={'fontsize': 12})
    # x y labels
    plt.xlabel('Epochs', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.ylabel('RMSE', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.plot(train_rmse, label='train_rmse', color="#333333")
    plt.legend(loc='best', fontsize=8)
    plt.savefig(save_path+'/train_rmse.jpg', dpi=500)
    # save the figure
    # plt.show()
    # close the figure
    plt.close()

    # alpha
    fig1 = plt.figure(figsize=(6, 4))
    ax1 = fig1.add_subplot(1, 1, 1)
    plt.title(str("Test Alpha"), fontweight="bold", color="#333333",
              pad=10, fontname='Times New Roman', fontdict={'fontsize': 10})
    # x y labels
    # plt.xlabel('Epochs',fontweight="bold", color="#333333", fontname = 'Times New Roman', fontdict={'fontsize':18}, labelpad=8)
    # plt.ylabel('Alpha',fontweight="bold", color="#333333", fontname = 'Times New Roman', fontdict={'fontsize':18}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.plot(np.sum(alpha1, 0), label="alpha", color="#333333")
    plt.legend(loc='best', fontsize=8)
    plt.savefig(save_path+'/alpha.jpg', dpi=500)
    # save the figure
    # plt.show()
    # close the figure
    plt.close()

    # alpha bar
    plt.title(str("Test Alpha"), fontweight="bold", color="#333333",
              pad=10, fontname='Times New Roman', fontdict={'fontsize': 10})
    # x y labels
    # plt.xlabel('Epochs',fontweight="bold", color="#333333", fontname = 'Times New Roman', fontdict={'fontsize':18}, labelpad=8)
    # plt.ylabel('Alpha',fontweight="bold", color="#333333", fontname = 'Times New Roman', fontdict={'fontsize':18}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.imshow(np.mat(np.sum(alpha1, 0)))
    plt.legend(loc='best', fontsize=8)
    plt.savefig(save_path+'/alpha11.jpg', dpi=500)
    # save the figure
    # plt.show()
    # close the figure
    plt.close()