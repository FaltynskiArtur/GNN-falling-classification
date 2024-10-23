import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Usuwanie wartości NaN
def remove_nan_values(data):
    # Zamiana wartości NaN na średnią wzdłuż osi dla współrzędnych X i Y
    for frame in range(data.shape[0]):  # Iteracja po każdej klatce
        for coord in range(data.shape[2]):  # Iteracja po współrzędnych X (0) i Y (1)
            nan_mask = np.isnan(data[frame, :, coord])  # Znalezienie NaN dla danej współrzędnej w danej klatce
            if np.any(nan_mask):
                # Zastąp NaN średnią wartością współrzędnej X lub Y (w danej klatce)
                data[frame, nan_mask, coord] = np.nanmean(data[frame, :, coord])
    return data

# Funkcja do aktualizacji wykresu dla każdej klatki

def update(frame_idx, data_2D, scat, lines, ax):
    ax.clear()  # Wyczyść poprzedni rysunek

    # Wyciągamy dane dla aktualnej klatki
    frame_data = data_2D[frame_idx]

    # Lista krawędzi (można dostosować do połączeń szkieletowych)
    edges = [
        (0, 1), (1, 8),  # Head to Neck
        (8, 2), (2, 3), (3, 4),  # Left arm
        (8, 5), (5, 6), (6, 7),  # Right arm
        (8, 9), (9, 11), (11, 12), (12, 13), (13, 14),  # Left leg
        (8, 10), (10, 15), (15, 16), (16, 17), (17, 18)  # Right leg
    ]

    # Rysowanie punktów (pozycji)
    xs, ys = frame_data[:, 0], frame_data[:, 1]
    scat = ax.scatter(xs, ys, c='r', marker='o', s=100)

    # Dodanie etykiet do punktów
    for i, label in enumerate(column_labels):
        ax.text(xs[i], ys[i], label, fontsize=8)

    # Rysowanie przerywanych krawędzi
    for (i, j) in edges:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], 'b--')

    # Odwrócenie osi Y
    ax.invert_yaxis()

    # Konfiguracja widoku
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_title(f"Szkielet 2D - Frame {frame_idx}")
    ax.set_aspect('equal', 'box')  # Zapewnienie proporcji osi

# Funkcja do stworzenia animacji
def animate_skeleton(path_to_mat_file):
    data_2D = load_2d_data_from_mat(path_to_mat_file)

    # Usunięcie wartości NaN
    data_2D = remove_nan_values(data_2D)

    # Inicjalizacja wykresu
    fig, ax = plt.subplots()
    scat = ax.scatter([], [])
    lines = []

    # Stworzenie animacji
    anim = FuncAnimation(
        fig, update, frames=len(data_2D), fargs=(data_2D, scat, lines, ax),
        interval=100, repeat=False  # Interval w ms, np. 100 ms to 10 fps
    )

    # Zapis do pliku GIF
    anim.save('skeleton_animation2D.gif', fps=10)

    # Wyświetlenie animacji na żywo
    plt.show()

# Użycie funkcji
path_to_mat_file = r"D:\FIR-Human\FIR-Human\BLOCK_3_FALL\FALLING_VOL_1\1_FORWARDS\1_STATIC\1\labels.mat"
animate_skeleton(path_to_mat_file)
