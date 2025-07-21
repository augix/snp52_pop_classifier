import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(conf_mat, outdir, current_epoch):
    # rows are expected, columns are predicted
    # normalize the confusion matrix so that the sum of the matrix is 100
    total = conf_mat.flatten().sum()
    conf_mat = conf_mat / total * 100
    # Calculate accuracy by summing diagonal elements (correct predictions) and dividing by total
    accuracy = np.sum(np.diag(conf_mat)) / 100
    # save the confusion matrix as a csv file
    np.savetxt(f'{outdir}/confusion_matrix_epoch{current_epoch}.csv', conf_mat, delimiter=',', fmt='%.4f')

    plt.figure(figsize=(12, 12))
    plt.imshow(conf_mat, cmap='Blues')
    plt.colorbar()
    plt.ylabel('Expected', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.title(f"Confusion Matrix - Epoch {current_epoch}, accuracy {accuracy:.3f}", fontsize=16)
    
    # Set integer ticks only
    n_classes = conf_mat.shape[0]
    plt.xticks(np.arange(n_classes))
    plt.yticks(np.arange(n_classes))
    
    # Add text annotations to show the values
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, f"{conf_mat[i, j]:.1f}", ha='center', va='center', color='black', fontsize=12)
    
    plt.savefig(f'{outdir}/confusion_matrix_epoch{current_epoch}.png')
    plt.close()
    return conf_mat

def plot_confusion_matrix_with_bubbles(conf_mat, outdir, current_epoch):
    # rows are expected, columns are predicted
    # normalize by total
    total = conf_mat.flatten().sum()
    conf_mat = conf_mat / total * 100
    # Calculate accuracy by summing diagonal elements (correct predictions) and dividing by total
    accuracy = np.sum(np.diag(conf_mat)) / 100

    # normalize by the sum of each row 
    conf_mat = conf_mat / conf_mat.sum(axis=1, keepdims=True) * 100

    # transpose the confusion matrix, so that rows are predicted and columns are expected
    conf_mat = conf_mat.T

    plt.figure(figsize=(12, 12))
    
    # Set up the plot
    n_classes = conf_mat.shape[0]
    x, y = np.meshgrid(np.arange(n_classes), np.arange(n_classes))
    
    # Scale bubble sizes - multiply by some factor to make bubbles visible but not overlapping
    sizes = conf_mat * 12  # Adjust multiplier as needed
    
    # replace 0 to n_classes
    # x[x == 0] = n_classes
    # y[y == 0] = n_classes

    plt.scatter(x.flatten(), y.flatten(), s=sizes.flatten(), alpha=.8, c='black', edgecolors='white')
    
    plt.xlabel('Expected', fontsize=20)
    plt.ylabel('Predicted', fontsize=20)
    plt.title(f"Confusion Matrix - Epoch {current_epoch}, accuracy {accuracy:.3f}", fontsize=20)
    
    # Set integer ticks
    ticks = np.arange(0,n_classes)
    tick_labels = [f'{i}' for i in ticks]
    # tick_labels[-1] = 'UN'
    plt.xticks(ticks, tick_labels, fontsize=16)
    plt.yticks(ticks, tick_labels, fontsize=16)
    
    # # Add text annotations to show the values
    # for i in range(n_classes):
    #     for j in range(n_classes):
    #         plt.text(j, i, f"{conf_mat[i, j]:.0f}", ha='center', va='center', color='gray', fontsize=12) if conf_mat[i, j] > 20 else None
    
    # Add gridlines to make it easier to read
    # plt.grid(True, linestyle='--', alpha=0.7)
    
    # expand the xlim and ylim so that the bubbles are not cut off
    plt.xlim(-1, n_classes)
    plt.ylim(-1, n_classes)
    plt.savefig(f'{outdir}/confusion_matrix_bubble_epoch{current_epoch}.png')
    plt.close()
    
    # save the confusion matrix as a csv file
    np.savetxt(f'{outdir}/confusion_matrix_bubble_epoch{current_epoch}.csv', conf_mat, delimiter=',', fmt='%.4f')
    return conf_mat

import os
def plot_predictions(outdir, input, target, pred, png_name='predictions.png'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cmap='rainbow'
    plt.figure(figsize=(12, 8))
    fig, axs = plt.subplots(3,1)
    axs[0].title.set_text('Input')
    axs[1].title.set_text('Target')
    axs[2].title.set_text('Prediction')
    x = input.flatten()
    y = target.flatten()
    vmin, vmax = min(x.min(), y.min()), max(x.max(), y.max())
    fig.colorbar(axs[0].imshow(input,cmap=cmap, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax ), ax=axs[0], shrink=0.6, orientation='vertical')
    fig.colorbar(axs[1].imshow(target,cmap=cmap, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax ), ax=axs[1], shrink=0.6, orientation='vertical')
    fig.colorbar(axs[2].imshow(pred,cmap=cmap, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax ), ax=axs[2], shrink=0.6, orientation='vertical')
    
    # turn off y-axis labels
    axs[0].set_yticks([])
    axs[1].set_yticks([])
    axs[2].set_yticks([])
    # tight layout
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, png_name))
    plt.close()

def plot_scatter(outdir, target, pred, title='Scatter Plot', png_name='scatter.png'):
    # calculate pearson correlation coefficient
    pearson_corr = np.corrcoef(target, pred)[0, 1]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.figure(figsize=(12, 8))
    # Add small random noise to break up overlapping points
    # noise_scale = (target.max() - target.min())/100
    noise_scale = 0.1
    target_jitter = target + np.random.normal(0, noise_scale, size=target.shape)
    # pred_jitter = pred + np.random.normal(0, noise_scale, size=pred.shape)
    plt.scatter(target_jitter, pred, alpha=0.5, s=0.5, c=target_jitter, cmap='jet')
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.colorbar(label='Target')
    plt.title(f'{title}, Pearson Corr: {pearson_corr:.3f}')
    plt.savefig(os.path.join(outdir, png_name))
    plt.close()
    print(f'Saved scatter plot to {os.path.join(outdir, png_name)}')