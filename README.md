
# Reconhecimento de Emoções com SSD MobileNetV2 🎭🎧

> **Atualização:** Este projeto agora utiliza o modelo **SSD MobileNetV2** para detecção de rostos em tempo real, conforme referências do professor. O modelo é carregado via **TensorFlow Hub**. A classificação de emoções continua sendo feita com o modelo treinado em FER-2013.

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

## 📁 Estrutura do Projeto

```
reconhecimento-faces/
├── train/                # imagens por classe para treino
├── test/                 # imagens por classe para teste
├── model.py              # define o modelo CNN
├── train.py              # treina o modelo com FER-2013 (imagens)
├── main.py               # detecta rosto com SSD MobileNetV2 e classifica emoção
├── verifica_cameras.py   # identifica índice da webcam
├── modelo_final.h5       # modelo treinado
├── requirements.txt      # bibliotecas usadas
└── README.md             # este arquivo
```

---

## 🚀 Como Executar

### 1. Instale as dependências
```bash
pip install -r requirements.txt
```

### 2. Prepare as pastas de imagens
Coloque `train/` e `test/` dentro da pasta `reconhecimento-faces/`, cada uma com subpastas por emoção:

```
train/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

### 3. Treine o modelo
```bash
python train.py
```

Isso vai gerar o `modelo_final.h5`.

### 4. Execute o sistema
```bash
python main.py
```

Ele usará a webcam, detectará seu rosto com SSD MobileNetV2, preverá a emoção e abrirá uma playlist relacionada.

---

## 📌 Créditos

- Modelo SSD MobileNetV2 via TensorFlow Hub
- Dataset: [FER-2013 - Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
