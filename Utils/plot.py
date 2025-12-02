import matplotlib.pyplot as plt

def plot_noise_vs_accuracy(noise_levels, accuracies, out_path):
    plt.figure(figsize=(8,6))

    # use discrete, actual noise values on the x-axis (no log scale)
    plotted_x = noise_levels

    plt.plot(plotted_x, accuracies, marker='o')
    # removed plt.xscale('log') to keep discrete actual values
    plt.xlabel('Noise Level (σ)')
    plt.ylabel('Accuracy (%)')
    plt.title('Effect of Noise on Model Accuracy')
    plt.grid(True)

    # force each noise level to appear as an x-tick and show original labels
    tick_labels = [f"{x:.8g}" for x in noise_levels]
    plt.xticks(plotted_x, tick_labels, rotation=45)

    plt.tight_layout()

    # save first, then show (so file is written even if GUI is closed)
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

    # show the plot (will block until closed)
    plt.show()
    plt.close()
    
    return

def main():
    # Example usage
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
    accuracies = [0.93373107, 0.93169099, 0.92997527, 0.92793533, 0.92556859, 0.92352822, 0.91985599, 0.91846736]

    out_path = './noise_vs_accuracy.png'
    
    plot_noise_vs_accuracy(noise_levels, accuracies, out_path)

main()