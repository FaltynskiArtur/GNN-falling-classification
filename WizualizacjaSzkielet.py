import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
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
        return mat['data_3D']  # Załadowanie danych
    else:
        raise ValueError("Brak danych 'data_3D' w pliku.")

# Funkcja do wizualizacji szkieletu
def visualize_skeleton_3D(data_3D, frame_idx=0):

    # Zakładamy, że 'data_3D' ma kształt (num_frames, num_joints, 3)
    frame_data = data_3D[frame_idx]  # Wybieramy dane dla jednego frame'a (klatki)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Lista krawędzi (można dostosować do połączeń szkieletowych)
    edges = [
        (0, 1), (1, 8),  # Head to Neck
        (8, 2), (2, 3), (3, 4),  # Left arm
        (8, 5), (5, 6), (6, 7),  # Right arm
        (8, 9), (9, 11), (11, 12), (12, 13), (13, 14),  # Left leg
        (8, 10), (10, 15), (15, 16), (16, 17), (17, 18)  # Right leg
    ]

    # Zamiana osi X z Y
    ys, xs, zs = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]
    ax.scatter(xs, ys, zs, c='r', marker='o', s=100)

    # Dodanie etykiet do punktów
    for i, label in enumerate(column_labels):
        ax.text(xs[i], ys[i], zs[i], label)

    # Rysowanie przerywanych krawędzi
    for (i, j) in edges:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], 'b--')  # 'b--' oznacza przerywaną niebieską linię

    # Konfiguracja widoku
    ax.set_xlabel('Y Label')  # Zmieniono na Y
    ax.set_ylabel('X Label')  # Zmieniono na X
    ax.set_zlabel('Z Label')  # Z pozostaje bez zmian
    ax.set_title(f"Szkielet 3D - Frame {frame_idx} ")
    # Zmiana pozycji kamery
    ax.view_init(elev=90, azim=90)  # Tutaj ustawiamy startowy widok kamery
    plt.show()

# Użycie funkcji
path_to_mat_file = r"D:\FIR-Human\FIR-Human\BLOCK_3_FALL\FALLING_VOL_1\1_FORWARDS\1_STATIC\1\labels.mat"
data_3D = load_3d_data_from_mat(path_to_mat_file)

# Wizualizacja szkieletu dla pierwszej klatki
visualize_skeleton_3D(data_3D, frame_idx=0)
