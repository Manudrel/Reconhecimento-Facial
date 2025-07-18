# Reconhecimento Facial em Tempo Real no Google Colab com YOLOv8

Este projeto demonstra como realizar **detecção de rostos em tempo real** utilizando um modelo YOLOv8 treinado com dataset do Roboflow, integrado à webcam no Google Colab.

---

## **Funcionalidades**

* Acessa a webcam diretamente pelo navegador via Colab.
* Treina um modelo YOLOv8 personalizado com dataset do Roboflow.
* Detecta rostos em tempo real, desenhando bounding boxes sobre as faces detectadas.
* Permite realizar testes com imagens estáticas após o treinamento.

---

## **Tecnologias e Bibliotecas Utilizadas**

* [Python 3](https://www.python.org/)
* [Google Colab](https://colab.research.google.com/)
* [OpenCV](https://opencv.org/)
* [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
* [Roboflow](https://roboflow.com/)
* [Javascript (para integração webcam-Colab)](https://developer.mozilla.org/en-US/docs/Web/API/Webcam_API)
* `numpy`, `matplotlib`, `PIL`, `io`, `html`, `contextlib`, `logging`, `sys`, `IPython.display`

---

## **Como Executar**

### 1. **Clone ou copie o notebook**

Utilize diretamente no Google Colab.

### 2. **Instale as dependências**

```python
!pip install -q ultralytics roboflow
```

### 3. **Configure a Webcam no Colab**

O código utiliza scripts Javascript para habilitar e capturar frames da webcam via navegador.

### 4. **Treine o modelo YOLOv8**

* O projeto usa um dataset do **Roboflow**. Para isso, configure sua `API_KEY` e as informações do dataset:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="SUA_API_KEY")
project = rf.workspace("seu_workspace").project("nome_do_projeto")
version = project.version(1)
dataset = version.download("yolov8")
```

* Treine o modelo:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
     data = "caminho_para_data.yaml",
     epochs = 20,
     imgsz = 640,
 )
```

### 5. **Execute o Loop Principal**

O loop captura frames da webcam, processa com o YOLOv8 e desenha bounding boxes nos rostos detectados em tempo real.

### 6. **Teste com Imagem Estática (Opcional)**

Após treinar o modelo, teste com uma imagem estática para verificar resultados:

```python
from google.colab import files
import matplotlib.pyplot as plt
import cv2

def mostrar(frame):
    imagem = cv2.imread(frame)
    if imagem is None:
        print(f"Erro: imagem '{frame}' não encontrada.")
        return
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.show()

# uploaded = files.upload()
# image_path = next(iter(uploaded))

# results = model(image_path)
# results[0].save(filename='predictions.jpg')

# mostrar('predictions.jpg')
```

---

## **Referências**

* **Tutorial Base (The AI Guy):**
  [How to Use Webcam In Google Colab for Images and Video (FACE DETECTION)](https://www.youtube.com/watch?v=xxx)

* **Notebook Base para as funcionalidades da câmera:**
  [Google Colab do The AI Guy](https://colab.research.google.com/drive/1QnC7lV7oVFk5OZCm75fqbLAfD9qBy9bw?usp=sharing)

* **YOLOv8 Documentation:**
  [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)

* **Roboflow Docs:**
  [Roboflow Documentation](https://docs.roboflow.com/)

---

## **Observações Importantes**

* Para rodar a webcam, é necessário permitir o acesso do navegador ao dispositivo.
* O Colab pode apresentar limitações em performance dependendo do navegador e do modelo utilizado.
* Substitua a `API_KEY` do Roboflow pela sua chave pessoal para acesso ao dataset.

---

## **Autor**

> Este projeto foi configurado e adaptado por **\[Emanuel Duarte]**, para estudos de visão computacional aplicada a detecção de rostos em tempo real.

---

