#Veamos un ejemplo concreto de una red neuronal que usa el paquete Keras R para aprender a clasificar los dígitos escritos a mano. El problema que intentamos resolver aquí es clasificar las imágenes en escala de grises de los dígitos escritos a mano (28 píxeles por 28 píxeles) en 10 categorías (números del 0 a 9). Usaremos el conjunto de datos MNIST, que consta de 60,000 imágenes de entrenamiento, más 10,000 imágenes de prueba, reunidas por el Instituto Nacional de Estándares y Tecnología (el NIST en MNIST) en la década de 1980. 

#El conjunto de datos MNIST viene precargado en Keras, en forma de listas `train` y` test`, cada una de las cuales incluye un conjunto de imágenes (`x`) y etiquetas asociadas (` y`):

library(keras)

mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
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

#Con este pequeño ejemplo puedes observar como contruir y entrenar una red neuronal para clasificar dígitos escritos a mano en menos de 20 líneas de código. 

#Veamos ahora como se realiza la regularización. En Keras, la regularización de peso se realiza  pasando instancias de regularizadores a las 
#capas como argumentos fundamentales.

library(keras)

imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

vectorize_sequences <- function(sequences, dimension = 10000) {
  # Create an all-zero matrix of shape (len(sequences), dimension)
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    # Sets specific indices of results[i] to 1s
    results[i, sequences[[i]]] <- 1
  results
}

# Our vectorized training data
x_train <- vectorize_sequences(train_data)
# Our vectorized test data
x_test <- vectorize_sequences(test_data)

# Our vectorized labels
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

original_model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

original_model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

#A continuación construimos una red más pequeña
smaller_model <- keras_model_sequential() %>% 
  layer_dense(units = 4, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 4, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

smaller_model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

#Y entrenamos las dos:
original_hist <- original_model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_test, y_test)
)

smaller_model_hist <- smaller_model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_test, y_test)
)

#Ahora comparamos lasfunciones de pérdida de las dos redes. Construimos una función para ello.
library(ggplot2)
library(tidyr)
plot_training_losses <- function(losses) {
  loss_names <- names(losses)
  losses <- as.data.frame(losses)
  losses$epoch <- seq_len(nrow(losses))
  losses %>% 
    gather(model, loss, loss_names[[1]], loss_names[[2]]) %>% 
    ggplot(aes(x = epoch, y = loss, colour = model)) +
    geom_point()
}

plot_training_losses(losses = list(
  original_model = original_hist$metrics$val_loss,
  smaller_model = smaller_model_hist$metrics$val_loss
))

#Vamos a estudiar ahora el efecto de la regularización. Añadimos regularización L2.
l2_model <- keras_model_sequential() %>% 
  layer_dense(units = 16, kernel_regularizer = regularizer_l2(0.001),
              activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, kernel_regularizer = regularizer_l2(0.001),
              activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

l2_model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

#`regularizer_l2(0.001)` significa que a cada coeficiente en la matriz de pesos de esa capa se le añade  `0.001` por su valor a la función de pérdida total de la red. Como esta penalización solo se añade en entrenamiento, la pérdida de la red es mayor en entrenamiento que en test.

l2_model_hist <- l2_model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_test, y_test)
)

plot_training_losses(losses = list(
  original_model = original_hist$metrics$val_loss,
  l2_model = l2_model_hist$metrics$val_loss
))

#Como se puede observar, el modelo con regularización L2 es menos sensible al sobreajuste, incluso con el mismo número de parámetros que un modelo sin regularización.

#Se pueden utilizar otros regularizadores. 


# L1 regularization
regularizer_l1(0.001)

# L1 and L2 regularization at the same time
regularizer_l1_l2(l1 = 0.001, l2 = 0.001)


#En Keras, puede introducir el dropout en una red a través de `layer_dropout ()`, que se aplica a la salida de la capa:
  
layer_dropout(rate = 0.5)

#Vamos a añadir dos capas de salida:

dpt_model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

dpt_model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

dpt_model_hist <- dpt_model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_test, y_test)
)

Vemos los resultados

plot_training_losses(losses = list(
  original_model = original_hist$metrics$val_loss,
  dpt_model = dpt_model_hist$metrics$val_loss
))


#Las redes neuronales convolucionales han tenido mucho éxito porque trabajan muy bien con imágenes. 
#En general, las redes neuronales convolucionales van a estar construidas con una estructura que contendrá 3 tipos distintos de capas:
  

#* Una capa convolucional, que es la que le da le nombre a la red.
#* Una capa de reducción o de pooling, la cual va a reducir la cantidad de parámetros al quedarse con las características más comunes.
#* Una capa clasificadora totalmente conectada, la cual nos va dar el resultado final de la red.

#Las redes neuronales convolucionales se distinguen de cualquier otra red neuronal en que utilizan un operación llamada convolución en alguna de sus capas en lugar de utilizar la multiplicación de matrices que se aplica generalmente. La operación de convolución recibe como entrada una imagen a la que aplica un filtro que devuelve un mapa de las características de la imagen original, reduciendo así el número de parámetros. La convolución se basa en las siguientes cuestiones:
  
#* Interacciones dispersas, ya que al aplicar un filtro de menor tamaño sobre la entrada original podemos reducir drásticamente la cantidad de parámetros y cálculos.
#* Parámetros compartidos entre los distintos tipos de filtros.
#* Si las entradas cambian, las salidas van a cambiar también en forma similar.

#Después de la capa convolucional, esta capa reduce en la dimensión que aunque puede conducir a una pérdida de información, también reduce los cáclulos y con ello evita el sobreajuste.

#La operación que se suele utilizar en esta capa es `max-pooling`, que divide a la imagen de entrada en un conjunto de rectángulos y, respecto de cada subregión, se va quedando con el máximo valor.

#Vamos a resolver ahora el mismo problema de identificación de dígitos. Recuerda que con una red neuronal densa obtenemos un porcentaje de acierto del 97.8%. 

#Vamos ahora a construir una red neuronal convolucional básica. Utilizaremos las capas `layer_conv_2d()` y `layer_max_pooling_2d()`. 

#La red neuronal convolucional recibe como entrada tensores con la siguiente forma `(altura_imagen, ancho_imagen, canal_imagen)`. Así configruramos la red neuronal convolucional para procesar entradas de tamaño `(28, 28, 1)`. Por tanto en la primera capa  `input_shape = c(28, 28, 1)` .

library(keras)

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")

#Veamos la arquitectura de la red.

summary(model)

#Para calcular el número de parámetros tenemos que aplicar la fórmula:
  
# total.params = (altura.filtro * ancho.filtro * numero.canales+1) * numero.filtros

  
#  El número de canales en una imagen en escala de grises es $1$, para una imagen a color, sería $3$, una por cada canal RGB.

#Así el número de parámetros de la primera capa es $(3*3*1+1)*32$ y el de la segunda $((3*3*32)+1)*64$. Fíjate que cada filtro de la primera capa se transforma en un canal de la segunda.

#La salida de `layer_conv_2d()` y `layer_max_pooling_2d()` es un 3D tensor de forma `(altura, ancho, canal)`. El ancho y la altura se reducen según se profundiza en la red. El número de canales se controla por el primer argumento que se pasa a `layer_conv_2d()` (32 o 64).

#El siguiente paso es alimentar la última capa (de tamaño `(3, 3, 64)`) en una red densa, pero como estas redes procesan 1D tensores, se deben transformar las salidas 3D en salidas 1D.

model <- model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

#La última capa tiene 10 neuronas porque el objetivo es realizar multicategoría, con 10 categorías.

summary(model)

#Es decir, cada salida con forma `(3, 3, 64)` se transforma en un vector de tamaño `(576)` antes de aplicar 2 capas densas.

#Finalmente vamos a realizar el entrenamiento.

mnist <- dataset_mnist()
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist

train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images / 255

test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images / 255

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(
  train_images, train_labels, 
  epochs = 5, batch_size=64
)

#Vamos a evaluar ahora el modelo construido.

results <- model %>% evaluate(test_images, test_labels)

results

#Como podemos observar pasamos de un porcentaje de acierto de 97.8% al 99%.