import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import os

# Lista dei video


def output_img(video_paths , frames  = [1, 180, 213, 246, 380]):

    # Caricare il primo video per ottenere il numero totale di frame
    cap = cv2.VideoCapture(video_paths[0])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Generare frame casuali (stessi per tutti i video)
    random_frames = frames
    print(f"Frame selezionati: {random_frames}")

    # Estrarre i frame e organizzarli in una lista
    frames_grid = []
    labels = []

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        video_frames = []  # Per ogni video, raccogliamo i frame
        
        # Estrarre il nome del video (senza percorso)
        video_name = os.path.basename(video_path)
        labels.append(video_name)
        
        for frame_idx in random_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Vai al frame specifico
            ret, frame = cap.read()

            if ret:
                frame = cv2.resize(frame, (304 * 8, 240 * 8))  # Ridimensiona per uniformità
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converti BGR → RGB per Matplotlib
                video_frames.append(frame)

        cap.release()

        if video_frames:  
            frames_grid.append(video_frames)  # Aggiungi i frame del video alla griglia

    # Invertire righe e colonne
    if frames_grid:
        # Trasporre la griglia per invertire righe e colonne
        transposed_grid = list(zip(*frames_grid))  # Invertire righe e colonne
        
        # Creare la griglia di immagini
        row_images = []
        for frames in transposed_grid:
            row_images.append(cv2.hconcat(frames))  # Concatena ogni colonna in una riga

        grid_image = cv2.vconcat(row_images)  # Concatena le righe per creare la griglia finale
        
        # Creare la figura
        fig, ax = plt.subplots(figsize=(25, 25))
        ax.imshow(grid_image)
        ax.axis("off")  # Rimuove gli assi
        
        # Aggiungere le etichette sopra le immagini
        for i, label in enumerate(labels):
            for j, frame in enumerate(transposed_grid[i]):
                # Posizionamento etichetta
                if j == 0:  # Etichetta centrata solo sulla prima immagine della colonna
                    x_position = (i + 0.5) * (grid_image.shape[1] / len(labels))  # Posizione orizzontale
                    ax.text(x_position, -50, label, fontsize=15, fontweight='bold', ha='center', va='bottom', color='black')

        plt.savefig("grid.png", bbox_inches='tight')
        plt.show()
