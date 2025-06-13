
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import webbrowser
from tensorflow.keras.models import load_model

def carregar_modelos():
    # Carregar o modelo de detecção SSD MobileNetV2 do TensorFlow Hub
    detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
    # Carregar modelo de emoções treinado
    classificador_emocao = load_model("modelo_final.h5")
    return detector, classificador_emocao

def detectar_rosto(frame, detector):
    # Preprocessar imagem
    img = cv2.resize(frame, (320, 320))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)[tf.newaxis, ...]

    # Executar detecção
    result = detector(input_tensor)
    result = {key: value.numpy() for key, value in result.items()}

    # Pegar a primeira detecção com score alto (pode ser rosto ou não)
    for i in range(min(10, len(result["detection_scores"]))):
        score = result["detection_scores"][i]
        if score < 0.5:
            continue
        box = result["detection_boxes"][i]
        h, w, _ = frame.shape
        y1, x1, y2, x2 = box
        (x1, y1, x2, y2) = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
    return None, None

def prever_emocao(face, model, emocoes):
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48)).astype("float32") / 255.0
    face_input = np.expand_dims(face_resized, axis=(0, -1))
    pred = model.predict(face_input)
    return emocoes[np.argmax(pred)]

def sugerir_playlist(emocao, emocoes_playlist, ultima_emocao):
    if emocao != ultima_emocao:
        print(f"Emoção detectada: {emocao}")
        link = emocoes_playlist.get(emocao)
        if link:
            webbrowser.open(link, new=2)
        return emocao
    return ultima_emocao

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
    ultima_emocao = None

    if not captura.isOpened():
        print("Erro ao acessar a webcam.")
        return

    print("Sistema iniciado. Pressione 'q' para sair.")

    while True:
        ret, frame = captura.read()
        if not ret:
            break

        rosto, coords = detectar_rosto(frame, detector)
        if rosto is not None:
            try:
                emocao = prever_emocao(rosto, modelo_emocao, emocoes)
                ultima_emocao = sugerir_playlist(emocao, emocoes_playlist, ultima_emocao)
                x1, y1, x2, y2 = coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, emocao, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            except Exception as e:
                print("Erro na previsão de emoção:", e)

        cv2.imshow("Reconhecimento de Emoção com SSD MobileNet", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
