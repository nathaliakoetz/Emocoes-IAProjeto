import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import webbrowser
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import platform

def limpar_terminal():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

def carregar_modelos():
    print("Carregando modelos... Por favor, aguarde.")
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
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
    return None, None

def prever_emocao(face, model, emocoes):
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    face_resized = cv2.equalizeHist(face_resized.astype("uint8"))
    face_resized = face_resized.astype("float32") / 255.0
    face_input = np.expand_dims(face_resized, axis=(0, -1))
    pred = model.predict(face_input)
    return emocoes[np.argmax(pred)]

def main():
    limpar_terminal()
    print("🎭 Sistema de Reconhecimento de Emoção Facial")
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
    limpar_terminal()
    captura = cv2.VideoCapture(0)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not captura.isOpened():
        print("Erro ao acessar a webcam.")
        return

    print("Pressione ESPAÇO para capturar sua expressão (ou 'q' para sair).")
    janela = "Captura de Expressão Facial"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

    rosto_detectado = None
    coordenadas = None
    emocao_atual = ""

    while True:
        ret, frame = captura.read()
        if not ret:
            print("Erro ao capturar imagem.")
            break

        rosto, coords = detectar_rosto(frame, detector)

        if rosto is not None:
            try:
                emocao = prever_emocao(rosto, modelo_emocao, emocoes)
                emocao_atual = emocao
                x1, y1, x2, y2 = coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, emocao, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                rosto_detectado = rosto
            except:
                pass

        cv2.imshow(janela, frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            captura.release()
            cv2.destroyAllWindows()
            return
        elif key == 32 and rosto_detectado is not None:
            break

    captura.release()
    cv2.destroyAllWindows()
    limpar_terminal()
    print("🧠 Processando imagem... Aguarde.")

    if rosto_detectado is not None:
        os.makedirs("capturas", exist_ok=True)
        agora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        caminho_arquivo = f"capturas/rosto_{agora}.png"
        cv2.imwrite(caminho_arquivo, rosto_detectado)

        print("🎯 Resultado da Análise de Emoção")
        print(f"🕒 Captura salva em: {caminho_arquivo}")
        print(f"🙂 Emoção detectada: {emocao_atual}")
        resposta = input(f"Deseja abrir a playlist para '{emocao_atual}' agora? (s/n): ").strip().lower()
        if resposta == 's':
            link = emocoes_playlist.get(emocao_atual)
            if link:
                webbrowser.open(link, new=2)
        else:
            print("👌 Playlist não aberta. Encerrando o sistema.")
    else:
        print("Nenhum rosto detectado.")

if __name__ == "__main__":
    main()