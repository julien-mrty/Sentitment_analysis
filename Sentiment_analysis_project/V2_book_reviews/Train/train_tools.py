import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Add this function to plot training results
def plot_training_results(logger, num_classes=5):
    # First plot the 5 main metrics
    fig, axis = plt.subplots(3, 2, figsize=(18, 12))

    plot_loss_records(logger, axis[0, 0])
    plot_metric_records(logger, 'accuracy', 'Train and Validation Accuracy', 'Accuracy', axis[0, 1], 'accuracy')
    plot_metric_records(logger, 'precision', 'Train and Validation Precision', 'Precision', axis[1, 0], 'macro avg',
                        'precision')
    plot_metric_records(logger, 'recall', 'Train and Validation Recall', 'Recall', axis[1, 1], 'macro avg', 'recall')
    plot_metric_records(logger, 'f1-score', 'Train and Validation F1-score', 'F1-score', axis[2, 0], 'macro avg',
                        'f1-score')
    #plot_learning_rate(logger, axis[2, 1])

    plt.tight_layout()
    plt.show()

    # Then plot the confusion matrices for Train and validation
    plot_confusion_matrix(logger)

    # Then plot the metrics for each class
    plot_metrics_per_class(logger, num_classes=num_classes)


def plot_loss_records(logger, axis):
    history = logger.get_history()

    train_loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = history['epoch']

    # Create the plot
    axis.plot(epochs, train_loss, marker='o', linestyle='-', color='b', label='Train loss')
    axis.plot(epochs, val_loss, marker='x', linestyle='-', color='r', label='Validation loss')
    axis.set_xlabel('Epochs')
    axis.set_ylabel('Loss values')
    axis.set_title('Train and Validation Loss Values Through Epochs')
    axis.legend()
    axis.grid(True)


def plot_learning_rate(logger, axis):
    history = logger.get_history()

    learning_rate = history['learning_rate']
    if not learning_rate: # Check if list is empty, lr have not been tracked from the beginning.
        return
    epochs = history['epoch']

    # Create the plot
    axis.plot(epochs, learning_rate, marker='o', linestyle='-', color='b', label='learning rate')
    axis.set_xlabel('Epochs')
    axis.set_ylabel('Learning rate')
    axis.set_title('Learning rate through epochs')
    axis.legend()
    axis.grid(True)


def plot_metric_records(logger, metric_name, title, ylabel, axis, report_key=None, report_subkey=None):
    try:
        history = logger.get_history()

        if report_subkey:
            train_values = logger.get_values_from_reports_with_inside_key("train_report", report_key, report_subkey)
            val_values = logger.get_values_from_reports_with_inside_key("val_report", report_key, report_subkey)
        else:
            train_values = logger.get_values_from_reports("train_report", metric_name)
            val_values = logger.get_values_from_reports("val_report", metric_name)

        epochs = history['epoch']

        axis.plot(epochs, train_values, marker='o', linestyle='-', color='b', label=f'Train {metric_name}')
        axis.plot(epochs, val_values, marker='x', linestyle='-', color='r', label=f'Validation {metric_name}')
        axis.set_xlabel('Epochs')
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend()
        axis.grid(True)
    except KeyError as e:
        print(f"Error: {e} not found in the logger's history.")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")


def plot_confusion_matrix(logger):
    history = logger.get_history()

    train_confusion_matrix = history['train_confusion_matrix']
    val_confusion_matrix = history['val_confusion_matrix']

    train_mean_confusion_matrix = np.sum(train_confusion_matrix, axis=0) / len(train_confusion_matrix)
    val_mean_confusion_matrix = np.sum(val_confusion_matrix, axis=0) / len(val_confusion_matrix)

    n_labels = len(train_mean_confusion_matrix[0])
    print("N LABELS in confusion matrix, ", n_labels)
    labels_indexes = list(range(1, n_labels + 1))

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))  # Adjust size depending on your needs

    # Plot the first confusion matrix
    sns.heatmap(train_mean_confusion_matrix, annot=False, cmap=plt.cm.Blues, xticklabels=labels_indexes, yticklabels=labels_indexes, ax=axes[0])
    axes[0].set_title('Train Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].tick_params(axis='y', rotation=0)

    # Plot the second confusion matrix
    sns.heatmap(val_mean_confusion_matrix, annot=False, cmap=plt.cm.Blues, xticklabels=labels_indexes, yticklabels=labels_indexes, ax=axes[1])
    axes[1].set_title('Validation Confusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='y', rotation=0)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_metrics_per_class(logger, num_classes=5):
    """
    Plots precision, recall, and F1-score for each class on the same plot.
    """
    history = logger.get_history()
    epochs = history['epoch']
    metrics = ['precision', 'recall', 'f1-score']

    for class_idx in range(num_classes):
        plt.figure(figsize=(10, 6))  # Create a new figure for each class

        for metric in metrics:
            # Fetch metric values for each class
            train_metric_values = logger.get_values_from_reports_with_inside_key("train_report", str(class_idx), metric)
            val_metric_values = logger.get_values_from_reports_with_inside_key("val_report", str(class_idx), metric)

            # Plot the metric on the same plot
            plt.plot(epochs, train_metric_values, marker='o', linestyle='-', label=f'Train {metric.capitalize()}')
            plt.plot(epochs, val_metric_values, marker='x', linestyle='--', label=f'Validation {metric.capitalize()}')

        # Add labels, title, legend, and grid
        plt.xlabel('Epochs')
        plt.ylabel('Metric Value')
        plt.title(f'Metrics for Class {class_idx}')
        plt.legend()
        plt.grid(True)
        plt.show()
