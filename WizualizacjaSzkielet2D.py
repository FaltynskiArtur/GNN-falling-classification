import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Etykiety kolumn (punkty ciała)
column_labels = [
    "Head top", "Head front", "LShoulder", "LElbow", "LWrist", "RShoulder",
    "RElbow", "RWrist", "Neck", "Left hip", "Right hip", "Left Knee",
    "Left Ankle", "Left heel", "Left toe", "Right Knee", "Right Ankle",
    "Right heel", "Right toe"
]

# Funkcja do załadowania danych z pliku .mat
def load_2d_data_from_mat(file_path):
    mat = scipy.io.loadmat(file_path)
    if 'data_2D' in mat:
        return mat['data_2D']  # Załadowanie danych 2D
    else:
        raise ValueError("Brak danych 'data_2D' w pliku.")

# Funkcja do wizualizacji szkieletu 2D
def visualize_skeleton_2D(data_2D, frame_idx=0):
    # Zakładamy, że 'data_2D' ma kształt (num_frames, num_joints, 2)
    frame_data = data_2D[frame_idx]  # Wybieramy dane dla jednego frame'a (klatki)

    fig, ax = plt.subplots()

    # Lista krawędzi (można dostosować do połączeń szkieletowych)
    edges = [
        (0, 1), (1, 8),  # Head to Neck
        (8, 2), (2, 3), (3, 4),  # Left arm
        (8, 5), (5, 6), (6, 7),  # Right arm
        (8, 9), (9, 11), (11, 12), (12, 13), (13, 14),  # Left leg
        (8, 10), (10, 15), (15, 16), (16, 17), (17, 18)  # Right leg
    ]

    # Rysowanie punktów (pozycji) z większym rozmiarem
    xs, ys = frame_data[:, 0], frame_data[:, 1]
    ax.scatter(xs, ys, c='r', marker='o', s=100)  # Zwiększenie rozmiaru punktów (s=100)

    # Dodanie etykiet do punktów
    for i, label in enumerate(column_labels):
        ax.text(xs[i], ys[i], label)

    # Rysowanie przerywanych krawędzi
    for (i, j) in edges:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], 'b--')  # 'b--' oznacza przerywaną niebieską linię

    # Odwrócenie osi Y, aby była naturalna
    ax.invert_yaxis()

    # Konfiguracja widoku
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_title(f"Szkielet 2D - Frame {frame_idx}")
    ax.set_aspect('equal', 'box')  # Zapewnienie proporcji osi
    plt.show()

# Użycie funkcji
path_to_mat_file = r"D:\FIR-Human\FIR-Human\BLOCK_3_FALL\FALLING_VOL_1\1_FORWARDS\1_STATIC\1\labels.mat"
data_2D = load_2d_data_from_mat(path_to_mat_file)

# Wizualizacja szkieletu dla pierwszej klatki
visualize_skeleton_2D(data_2D, frame_idx=0)
