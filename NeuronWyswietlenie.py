import os
import scipy.io
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Sprawdzanie dostępności GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ścieżka do przeszukania
path_to_search = r'D:/FIR-Human/FIR-Human'

# Import funkcji przetwarzania danych z osobnego modułu
from data_processing import prepare_data, create_graph, augment_data, extract_data_from_mat


# Import modelu GCN z osobnego modułu
from gcn_model import GNNWithResiduals


def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path='best_model.pth', activation_fn=F.relu):
    model = GNNWithResiduals(activation_fn)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model


def process_video(video_path, model, joints_data):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(joints_data):
            joints = joints_data[frame_idx]
            # Visualize joints on the frame
            for joint in joints:
                x, y, _ = joint
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Here, you should extract features from the video frame as per your model's requirement
        # Assuming you have a function to convert the frame to your required format, e.g., `convert_frame_to_graph`
        graph = convert_frame_to_graph(joints)
        graph = graph.to(device)

        with torch.no_grad():
            out = model(graph)
            pred = out.argmax(dim=1).item()

        # Display the prediction on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Prediction: {pred}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the frame with the prediction and joints
        cv2.imshow('Video', frame)

        frame_idx += 1

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def convert_frame_to_graph(joints):
    # Convert the joints data to a graph
    num_nodes = joints.shape[0]
    edge_index = torch.tensor([
        [i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j
    ], dtype=torch.long).t().contiguous()
    x = torch.tensor(joints, dtype=torch.float)
    graph = Data(x=x, edge_index=edge_index)
    return graph


if __name__ == "__main__":
    model = load_model('NAJLEPSZY985.pth', F.relu)
    video_path = r"D:\FIR-Human\FIR-Human\BLOCK_3_FALL\FALLING_VOL_4\2_BACKWARDS\2_DYNAMIC\1\FinalVideo.avi"
    joints_data_path = r"D:\FIR-Human\FIR-Human\BLOCK_3_FALL\FALLING_VOL_4\2_BACKWARDS\1_STATIC\1\labels.mat"

    # Load joints data from .mat file
    joints_data = extract_data_from_mat(joints_data_path)

    if joints_data is not None:
        process_video(video_path, model, joints_data)
    else:
        print("Could not load joints data.")
