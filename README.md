# REDES-NEURONALES-EN-CLASE

In this project we train a neural network to classify if an image is a dog or a cat.

Before training the model we reshape the images (taken from https://www.microsoft.com/en-us/download/details.aspx?id=54765) so the size is 200*200. This is done in the notebook DogCatResize.ipynb.

Then, we train the neural network using Tensorflow in the notebook Cat_Dog_Model.ipynb.

Para este trabajo trajimos las imagenes de perros y gatos en archivos zip separados.
Creamos una carpeta Pets y allí pusimos la carpeta perro y la carpeta gatos.
Rediseñamos el tamaño de las imagenes a un punto que aún perdiendo calidad se pudiera diferenciar si es un perro o un gato
Detectamos y eliminamos las imágenes que venían con error
Etiquetamos las imagenes como perro o gato según corresponde.
Creamos un dataset de entrenamiento y el de validación usando los siguientes parámetros
IMAGE_WITDH = 200
IMAGE_HEIGHT = 200
BATCH_SIZE = 32 #lote
Los resultados obtenidos fueron:
Total params: 34,765,761
Trainable params: 34,764,289
Non-trainable params: 1,472
salvamos el modelo
creamos una función que usando el modelo creado devuelve si la imagen es perro o gato
