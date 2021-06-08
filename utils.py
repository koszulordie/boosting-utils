import matplotlib.pyplot as plt


def plot_learning(learning_curve, eval_metric='rmse'):

    fig, ax = plt.subplots()
    train = learning_curve['validation_0'][eval_metric]
    test = learning_curve['validation_1'][eval_metric]
    ax.plot(train, color='blue')
    ax.plot(test, color='red')
    ax.set_xlabel('no. estimators')
    ax.set_ylabel(eval_metric)
    ax.set_title('Learning Curve')
    plt.show()
