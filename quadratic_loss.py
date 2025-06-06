from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "custom",
    [
        "#4a90e2",
        "#f8e71c",
        "#ff6b6b",
    ],
)

DATA = [(150, 200), (200, 600), (260, 500)]


def calculate_loss(w, b):
    return np.mean(np.array([(w * x + b - y) ** 2 for x, y in DATA]), axis=0)


def main():
    plt.rcParams["font.size"] = 16

    ws, bs = np.meshgrid(np.linspace(-10, 10, 400), np.linspace(-1000, 1000, 400))
    z = calculate_loss(ws, bs)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(ws, bs, z, cmap=CUSTOM_CMAP)
    ax.set_xlabel("$w$", fontsize=16)
    ax.set_ylabel("$b$", fontsize=16)
    ax.set_zlabel("$J(w,b)$", fontsize=16)
    ax.set_box_aspect(aspect=None, zoom=0.95)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
