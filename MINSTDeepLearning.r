#' ---
#' title: "Métodos Basados en el conocimiento Aplicados a la Empresa - Aprendizaje profundo con MNIST"
#' author: "Carlos García Gutiérrez (UO139393)"
#' output: pdf_document
#' ---

library(keras)

#' Obtenemos el dataset **MNIST**
mnist <- dataset_mnist()

#' Definimos una semilla con los dígitos del DNI y generamos una secuencia aleatoria con un tamaño
#' de la mitad del de la lista de imágenes/etiquetas
set.seed(DNI_SEED)
sample_array <- sample.int(nrow(mnist$train$x), size = floor(.50 * nrow(mnist$train$x)))

#' Obtenemos la mitad de las imágenes/etiquetas para entrenar, el conjunto de test es el completo
train_images <- mnist$train$x[sample_array,,]
train_labels <- mnist$train$y[sample_array]
test_images <- mnist$test$x
test_labels <- mnist$test$y

#Veamos la dimensión: es un tensor de 3 dimensiones
length(dim(train_images))

dim(train_images)

#El tipo de datos es

typeof(train_images)

#Podemos acceder a una imagen del conjunto de entrenamiento

digit <- train_images[5,,]
plot(as.raster(digit, max = 255))


#`train_images` y `train_labels` forman el _conjunto de entrenamiento_ a partir de los cuales se realizará el aprendizaje.
#El modelo se probará con el _conjunto de test_, `test_images` y` test_labels`. Las imágenes se codifican en arrays 3D, como se puede observar utilizando la función `str()` y las etiquetas son un conjunto de dígitos 1D, que van de 0 a 9. Hay una correspondencia de uno a uno entre las imágenes y las etiquetas.

#La función R `str ()` es una forma conveniente de echar un vistazo rápido a la estructura de una matriz. Vamos a usarlo para echar un vistazo a los datos de entrenamiento:
  
 str(train_images)

 str(train_labels)

#El flujo de trabajo será el siguiente: primero alimentaremos la red neuronal con los datos de entrenamiento, `train_images` y `train_labels`. La red aprenderá a asociar imágenes y etiquetas. Finalmente, le pediremos a la red que produzca predicciones para `test_images`, y verificaremos si estas predicciones coinciden con las etiquetas de `test_labels`.

network <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>% 
  layer_dense(units = 10, activation = "softmax")

summary(network)

#En las líneas anteriores se define la arquitectura de la red neuronal como una secuencia de dos capas denso-conectadas. La primera consta de 512 unidades de salida, asociadas a las entradas que de cada uno de los pixels (28*28) de la imagen. La segunda capa tiene 10 unidades, cada una de ellas asociada a un dígito (del 0 al 9). La función de activación _softmax_ hace que se retorne un array de 10 valores de probabilidad, cuya suma obviamente debe ser 1. 

#Una vez que la arquitectura de la red está definida, es preciso definir el resto de ingredientes. En las siguientes líneas indicamos como va a ser el entrenamiento, en concreto como se optimizará la red, cual es la función de pérdida, y cual será la métrica de evaluación. En este caso, la función de pérdida, es decir, la función a minimizar, es __categorical_crossentropy__. El optimizador, es decir el mecanismo de ajuste de los pesos es `rmsprop`, es decir __descenso del gradiente mini-batch__ (seleccionando solo un conjunto de pesos). 

network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# El operador %>% pasa el objeto que está a su izquierda como 
# primer argumento de la función de su derecha

#Antes de entrenar, reconfiguramos los datos a la forma que la red espera y __escalamos para que todos los valores estén en el intervalo__ `[0, 1]`. Anteriormente, nuestras imágenes de entrenamiento, por ejemplo, se almacenaban en una matriz de forma `(60000, 28, 28)` de tipo entero con valores en el intervalo `[0, 255]`. Lo transformamos matriz de orden `(60000, 28 * 28)` con valores entre 0 y 1.


train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255

#El conjunto de entrenamiento está almacenado en un 2-tensor (una matriz) de dimensión (60000, 784) 
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

#Finalmente ya podemos entrenar la red, que en Keras se hace con la función `fit`. En este caso la red se entrena seleccionado subconjuntos de 128 ejemplos (_batches_) sobre los que se entrena 5 veces (__epoch__). En cada iteración, la red computará los gradientes de los pesos con respecto a la función de pérdida pérdida sobre cada _batch_, y actualizará los pesos
#en consecuencia. Después de estas 5 iteraciones, la red habrá realizado 2,345 gradientes.
#actualizaciones (469 por _epoch_), y la pérdida de la red será lo suficientemente baja como para que La red será capaz de clasificar los dígitos escritos a mano con alta precisión.

network %>% fit(train_images, train_labels, epochs = 5, batch_size = 128)

#Se muestra el valor de la función de pérdida de la red y los datos de entrenamiento sobre los datos de entrenamiento.
#El porcentaje de acierto en entrenamiento es de 98.9% en los datos de entrenamiento. Comprobemos que nuestro modelo también funciona bien en el conjunto de prueba:
  
metrics <- network %>% evaluate(test_images, test_labels, verbose = 0)
metrics