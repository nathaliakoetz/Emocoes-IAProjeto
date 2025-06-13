import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import webbrowser
from tensorflow.keras.models import load_model

def carregar_modelos():
    detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
    classificador_emocao = load_model("modelo_final.h5")
    return detector, classificador_emocao

def detectar_rosto(frame, detector):
    img = cv2.resize(frame, (320, 320))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)[tf.newaxis, ...]
    result = detector(input_tensor)
    result = {key: value.numpy() for key, value in result.items()}

    for i in range(len(result["detection_scores"])):
        score = result["detection_scores"][i]
        if isinstance(score, np.ndarray):
            score = float(score[0]) if score.shape else float(score)
        if score < 0.5:
            continue
        box = result["detection_boxes"][i].flatten()[:4]
        y1, x1, y2, x2 = box
        h, w, _ = frame.shape
        x1, y1, x2, y2 = map(int, [x1 * w, y1 * h, x2 * w, y2 * h])
        return frame[y1:y2, x1:x2]
    return None

def prever_emocao(face, model, emocoes):
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48)).astype("float32") / 255.0
    face_input = np.expand_dims(face_resized, axis=(0, -1))
    pred = model.predict(face_input)
    return emocoes[np.argmax(pred)]

def sugerir_playlist(emocao, emocoes_playlist):
    print(f"Emoção detectada: {emocao}")
    link = emocoes_playlist.get(emocao)
    if link:
        webbrowser.open(link, new=2)

def main():
    emocoes = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro']
    emocoes_playlist = {
        'Raiva': 'https://open.spotify.com/playlist/37i9dQZF1DX3YSRoSdA634',
        'Nojo': 'https://open.spotify.com/playlist/37i9dQZF1DWU0ScTcjJBdj',
        'Medo': 'https://open.spotify.com/playlist/37i9dQZF1DX2pSTOxoPbx9',
        'Feliz': 'https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC',
        'Triste': 'https://open.spotify.com/playlist/37i9dQZF1DWVrtsSlLKzro',
        'Surpreso': 'https://open.spotify.com/playlist/37i9dQZF1DX1g0iEXLFycr',
        'Neutro': 'https://open.spotify.com/playlist/37i9dQZF1DX4WYpdgoIcn6'
    }

    detector, modelo_emocao = carregar_modelos()
    captura = cv2.VideoCapture(0)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not captura.isOpened():
        print("Erro ao acessar a webcam.")
        return

    print("Pressione ESPAÇO para capturar sua expressão (ou 'q' para sair).")

    janela = "Captura de Expressão Facial"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = captura.read()
        if not ret:
            print("Erro ao capturar imagem.")
            break

        cv2.imshow(janela, frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 32:  # tecla espaço
            rosto = detectar_rosto(frame, detector)
            if rosto is not None:
                emocao = prever_emocao(rosto, modelo_emocao, emocoes)
                sugerir_playlist(emocao, emocoes_playlist)
            else:
                print("Nenhum rosto detectado.")
            break

    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()