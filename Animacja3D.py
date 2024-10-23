import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Etykiety kolumn (punkty ciała)
column_labels = [
    "Head top", "Head front", "LShoulder", "LElbow", "LWrist", "RShoulder",
    "RElbow", "RWrist", "Neck", "Left hip", "Right hip", "Left Knee",
    "Left Ankle", "Left heel", "Left toe", "Right Knee", "Right Ankle",
    "Right heel", "Right toe"
]

# Funkcja do załadowania danych z pliku .mat
def load_3d_data_from_mat(file_path):
    mat = scipy.io.loadmat(file_path)
    if 'data_3D' in mat:
        return mat['data_3D']  # Załadowanie danych 3D
    else:
        raise ValueError("Brak danych 'data_3D' w pliku.")

# Funkcja do usuwania NaN i Inf z danych
def remove_nan_and_inf(data):
    mask = np.isfinite(data)
    data[~mask] = 0  # Zamiana NaN i Inf na 0 (lub można użyć innej wartości)
    return data

# Funkcja do aktualizacji wykresu dla każdej klatki
def update(frame_idx, data_3D, scat, lines, ax):
    ax.cla()  # Wyczyść poprzedni rysunek

    # Wyciągamy dane dla aktualnej klatki
    frame_data = data_3D[frame_idx]

    # Usunięcie NaN i Inf z danych
    frame_data = remove_nan_and_inf(frame_data)

    # Lista krawędzi (można dostosować do połączeń szkieletowych)
    edges = [
        (0, 1), (1, 8),  # Head to Neck
        (8, 2), (2, 3), (3, 4),  # Left arm
        (8, 5), (5, 6), (6, 7),  # Right arm
        (8, 9), (9, 11), (11, 12), (12, 13), (13, 14),  # Left leg
        (8, 10), (10, 15), (15, 16), (16, 17), (17, 18)  # Right leg
    ]

    # Rysowanie punktów (pozycji)
    xs, ys, zs = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]
    scat = ax.scatter(xs, ys, zs, c='r', marker='o', s=100)

    # Dodanie etykiet do punktów
    for i, label in enumerate(column_labels):
        ax.text(xs[i], ys[i], zs[i], label, fontsize=8)

    # Rysowanie przerywanych krawędzi
    for (i, j) in edges:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], 'b--')

    # Ustawianie limitów osi tylko dla skończonych wartości
    if np.isfinite(xs).all() and np.isfinite(ys).all() and np.isfinite(zs).all():
        ax.set_xlim(np.min(xs), np.max(xs))
        ax.set_ylim(np.min(ys), np.max(ys))
        ax.set_zlim(np.min(zs), np.max(zs))

    # Konfiguracja widoku
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(f"Szkielet 3D - Frame {frame_idx}")

    # Zmiana pozycji kamery
    ax.view_init(elev=90, azim=90)  # Tutaj ustawiamy startowy widok kamery
# Funkcja do stworzenia animacji
def animate_skeleton_3D(path_to_mat_file):
    data_3D = load_3d_data_from_mat(path_to_mat_file)

    # Usunięcie NaN i Inf z całych danych
    data_3D = remove_nan_and_inf(data_3D)

    # Inicjalizacja wykresu
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter([], [], [])
    lines = []

    # Stworzenie animacji
    anim = FuncAnimation(
        fig, update, frames=len(data_3D), fargs=(data_3D, scat, lines, ax),
        interval=100, repeat=False  # Interval w ms, np. 100 ms to 10 fps
    )

    anim.save('skeleton_animation.gif', fps=10)

    # Wyświetlenie animacji na żywo
    plt.show()

# Użycie funkcji
path_to_mat_file = r"D:\FIR-Human\FIR-Human\BLOCK_3_FALL\FALLING_VOL_1\1_FORWARDS\1_STATIC\1\labels.mat"
animate_skeleton_3D(path_to_mat_file)
