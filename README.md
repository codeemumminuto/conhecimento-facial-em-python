# Conhecimento-Facial-em-Python 💻
Projeto criado a partir de um mini curso da dio.me (Digital Inovation One); Inteligência artificial / Conhecimento Facial / Detecção de Imagens

## Sistema de detecção de faces humanas
Nosso sistema deve ser capaz de detectar a região que representa a face, dando suporte para o sistema de classificação reconhecer a pessoa em questão.

#### Linguagens e Ferramentas utilizadas
 - Python
 - Google Colab
 
#### Bibliotecas e requerimentos necessários
```python
import imutils #Utilizada para redimencionamento e rotação da imagem
import numpy as np #Responsável para trabalhos matemáticos como vetores e matrizes das imagens
import cv2 #(OpenCv) Utilizado para ajudar no processo de detecção da face
from google.colab.patches import cv2_imshow #Dependencia do google colab para trabalhar com imagens
from IPython.display import display, Javascript #Bibliotecas necessárias para trabalhar com a leitura da webcam
from google.colab.output import eval_js #Dependencia do google colab para trabalhar com leitura da webcam
from base64 import b64decode #Utilizada para codificar dados binários
```
### Iniciando o código
#### Parte 1 - Leitura da webcam
O código abaixo é responsável pela leitura da webcam, pois estamos utilizado-a como entrada de imagem para detectar nossa face.

```python
def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename
```
#### Parte2 - Tratamento da imagem
O código baixo é responsável por chamar a nossa função criada anteriormente. Ela irá fazer com que a webcam abra e mostre sua imagem na tela 📷

```python
image_file = take_photo()
```
O código abaixo é responsável por redimencionar a imagem da webcam, no nosso caso escolhemos que a imagem fosse redimencionada para largura máxima de 400 pixel.

```python
image = cv2.imread(image_file)
image = imutils.resize(image, width=400)
(h, w) = image.shape[:2]
print(w,h)
cv2_imshow(image) #"Imprime" a imagem redimencionada na tela
```
#### Parte3 - Baixando e carregando os modelos
O detector de face em Deep Learning do OpenCV é baseado na estrutura Single Shot Detector (SSD) com uma rede base ResNet. A rede é definida e treinada usando o [Caffe Deep Learning framework](https://caffe.berkeleyvision.org/)

Baixe o modelo de detecção de rosto pré-treinado, composto por dois arquivos:

A definição de rede (deploy.prototxt) e os pesos aprendidos (res10_300x300_ssd_iter_140000.caffemodel)

Utilizando as linhas de código abaixo no google colab

```python
!wget -N https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
!wget -N https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

Estamos utilizandos esses arquivos, pois eles já estão treinados com o fim que nós queremos (detectar faces);
Agora vamos carregar o modelo de rede de detecção facial pré-treinado do disco:

```python
print("[INFO] loading model...")
prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)
```
Use a função dnn.blobFromImage para construir um blob de entrada redimensionando a imagem para 300x300 pixels fixos e normalizando-a:

```python
image = imutils.resize(image, width=400)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
```
Computar os objetos detectados na imagem em busca de uma face:

```python
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()
```
Loop para as detecções e desenhe caixas ao redor dos rostos detectados:

```python
for i in range(0, detections.shape[2]):

	# extrair a probabilidade associada à previsão
	confidence = detections[0, 0, i, 2]

	# filtra detecções fracas garantindo que a "confiança" seja
	# maior que o limite mínimo de confiança
	if confidence > 0.5: #Nossa detecção deve ter no mínimo 50% de certeza
		# calcula as coordenadas (x, y) da caixa delimitadora do objeto
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		# desenha a caixa delimitadora da face junto com a probabilidade associada
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
```

#### Parte4 - Obtendo o resultado
Agora, precisamos apenas gerar o resultado na tela com o código abaixo:

```python
cv2_imshow(image)
```
##### Esse resultado exibirá o nosso rosto dentro de um retângulo, junto à percentagem de certeza que é uma face humana, dessa forma:

![imagem](https://i.ibb.co/2MxRj3T/download1.png)
