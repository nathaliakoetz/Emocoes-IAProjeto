
# Reconhecimento de Emoções  🎭🎧

> Este projeto agora utiliza o modelo **SSD MobileNetV2** para detecção de rostos em tempo real. O modelo é carregado via **TensorFlow Hub**. A classificação de emoções é feita com o modelo treinado em FER-2013.

---

## 🔍 Funcionalidades

- Detecção de rostos com SSD MobileNetV2 via TensorFlow Hub
- Reconhecimento de emoções com CNN treinada no dataset FER-2013
- Recomendação de playlists do Spotify conforme a emoção detectada
- Uso de webcam para funcionamento em tempo real

---

## 🎯 Emoções Reconhecidas

- Raiva
- Nojo
- Medo
- Feliz
- Triste
- Surpreso
- Neutro

---

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

---

## 🧑‍💻 **Autores**  

<a href="https://github.com/nathaliakoetz"><img src="https://github.com/nathaliakoetz.png" width="100" height="100"></a>
<a href="https://github.com/gwacosta"><img src="https://github.com/gwacosta.png" width="100" height="100"></a>

