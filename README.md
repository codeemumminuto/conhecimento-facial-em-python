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
cv2_imshow(image)
```
