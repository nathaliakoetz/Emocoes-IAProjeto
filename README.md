
# Reconhecimento de Emoções  🎭🎧

> Este projeto utiliza o modelo **SSD MobileNetV2** para detecção de rostos em tempo real. O modelo é carregado via **TensorFlow Hub**. A classificação de emoções é feita com o modelo treinado em FER-2013.

## 🔍 Funcionalidades

- Detecção de rostos com SSD MobileNetV2 via TensorFlow Hub
- Reconhecimento de emoções com CNN treinada no dataset FER-2013
- Recomendação de playlists do Spotify conforme a emoção detectada
- Uso de webcam para funcionamento em tempo real

## 🎯 Emoções Reconhecidas

| ![Raiva](https://em-content.zobj.net/thumbs/240/apple/354/pouting-face_1f621.png) <br> **Raiva** | ![Nojo](https://em-content.zobj.net/thumbs/240/apple/354/nauseated-face_1f922.png) <br> **Nojo** | ![Medo](https://em-content.zobj.net/thumbs/240/apple/354/fearful-face_1f628.png) <br> **Medo** | ![Feliz](https://em-content.zobj.net/thumbs/240/apple/354/smiling-face-with-smiling-eyes_1f60a.png) <br> **Feliz** | ![Triste](https://em-content.zobj.net/thumbs/240/apple/354/crying-face_1f622.png) <br> **Triste** | ![Surpreso](https://em-content.zobj.net/thumbs/240/apple/354/face-with-open-mouth_1f62e.png) <br> **Surpreso** | ![Neutro](https://em-content.zobj.net/thumbs/240/apple/354/expressionless-face_1f611.png) <br> **Neutro** |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|


## 🚀 Como Executar

### 1. Instale as dependências
```bash
pip install -r requirements.txt
```

### 2. Treine o modelo
```bash
python train.py
```

Isso vai gerar o `modelo_final.h5`.

### 3. Execute o sistema
```bash
python main.py
```

Ele usará a webcam, detectará seu rosto com SSD MobileNetV2, preverá a emoção e abrirá uma playlist relacionada.

## 🧑‍💻 **Autores**  

<a href="https://github.com/nathaliakoetz"><img src="https://github.com/nathaliakoetz.png" width="100" height="100"></a>
<a href="https://github.com/gwacosta"><img src="https://github.com/gwacosta.png" width="100" height="100"></a>

