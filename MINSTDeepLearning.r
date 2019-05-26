#' ---
#' title: "Resolución de un problema de clasificación con aprendizaje profundo utilizando un subconjunto del conjunto MNIST"
#' author: "Carlos García Gutiérrez (UO139393)"
#' date:
#' output: pdf_document
#' ---

#' **Introducción**
#'
#' La ejecución de esta práctica consta de las siguientes partes:  
#' - Cargar en memoria los datos a utilizar  
#' - Crear, entrenar y comparar los resultados de varias configuraciones de redes densas
#' - Añadir regularización a las redes anteriores y comparar los resultados 
#' - Crear, entrenar y comparar los resultados de varias configuraciones de redes convolucionales  

#'  
#' **Carga de datos en memoria**

library(keras)

#' Obtenemos el dataset MNIST
mnist <- dataset_mnist()

#' Definimos una semilla con los dígitos del DNI y generamos una secuencia aleatoria con un tamaño
#' de la mitad del de la lista de imágenes/etiquetas
set.seed(53540153)
sample_array <- sample.int(nrow(mnist$train$x), size = floor(.10 * nrow(mnist$train$x)))
#PONER AL 50% ANTES DE ENTREGAR!!!

#' Obtenemos la mitad de las imágenes/etiquetas para entrenar; el conjunto de test es el completo
train_images <- mnist$train$x[sample_array,,]
train_labels <- mnist$train$y[sample_array]
test_images <- mnist$test$x
test_labels <- mnist$test$y

#' Se reordenan los datos para poder ser usados como entrada de las redes neuronales y se escalan los 
#' valores RGB de las imágenes para que estén en el intervalo [0, 1], asimismo se transforman los  
#' las etiquetas a valores binarios, según el dígito que representen

train_images <- array_reshape(train_images, c(nrow(train_images), 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(nrow(test_images), 28 * 28))
test_images <- test_images / 255
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

#'  
#' **Redes neuronales densas**

#'  
#' Vamos a crear tres redes neuronales densas: con dos capas (entrada y salida), con tres capas  
#' (igual que la anterior pero añadiendo una capa oculta) y con cuatros capas (igual que la primera  
#' pero añadiendo dos capas ocultas)  
#'   
#' Las capa de entrada contiene una neurona por cada pixel (28 x 28) y estas se activan utilizando  
#' la función ReLU, que es la adecuada para la escala de grises de las imágenes  
#'   
#' La capa de salida tiene 10 neuronas, que son las necesarias para las 10 categorías a considerar  
#' (valores del 0 al 9), en este caso la función de activación es la adecuada para problemas de  
#' clasificación múltiple de una etiqueta (como es nuestro caso)  
#'   
#' Evidentemente, para un problema tan sencillo, no serían necesarias redes con tantas capas, pero se  
#' ha decidido utilizar estas configuraciones para ilustrar el problema del sobreajuste e intentar  
#' minimizarlo posteriormente mediante la regularización  
#'   
#' Para todas las redes, la función a minimizar es "categorical_crossentropy", que es la adecuada para  
#' problemas de clasificación múltiple de una etiqueta. La optimización estará basada en el descenso  
#' del gradiente utilizando solo un conjunto de los pesos, según lo visto en clase

dense_network_2layers <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>% 
  layer_dense(units = 10, activation = "softmax")

dense_network_2layers %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

summary(dense_network_2layers)

dense_network_3layers <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

dense_network_3layers %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

summary(dense_network_3layers)

dense_network_4layers <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

dense_network_4layers %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

summary(dense_network_4layers)

#' Se realiza el entrenamiento de las redes (utilizando cinco iteraciones)

dense_network_2layers %>% fit(train_images, train_labels, epochs = 5, batch_size = 128)
dense_network_3layers %>% fit(train_images, train_labels, epochs = 5, batch_size = 128)
dense_network_4layers %>% fit(train_images, train_labels, epochs = 5, batch_size = 128)

#' Se obtienen y se muestran los resultados
  
metrics_dn_2_layers <- dense_network_2layers %>% evaluate(test_images, test_labels, verbose = 0)
metrics_dn_3_layers <- dense_network_3layers %>% evaluate(test_images, test_labels, verbose = 0)
metrics_dn_4_layers <- dense_network_4layers %>% evaluate(test_images, test_labels, verbose = 0)
metrics_dn_2_layers
metrics_dn_3_layers
metrics_dn_4_layers

#' Se puede observar que añadir una capa oculta a la red mejora los resultados, pero añadir una segunda  
#' capa oculta los vuelve a empeorar; esto era de esperar ya que las redes más profundas producen un  
#' sobreajuste al modelo, podrías decir entonces, que una configuración con una capa oculta es la que  
#' mejor se ajusta a este problema y este conjunto de datos.  
#'   
#' Se podría aunmentar también el sobreajuste haciendo las capas ocultas más grandes o añadiendo más  
#' iteraciones, pero con las tres redes utilizadas queda conseguido perfectamente el objetivo de  
#' ilustrar como una red compleja tiende a sobreajustar.   

#'  
#' **Regularización**

#'   
#' La regularización es una técnica que intenta mitigar el sobreajuste de las redes neuronales basándose  
#' en el principio de que, a igualdad de condiciones, se debe utilizar el modelo más sencillo. Para ello  
#' se intenta limitar la complejidad de la red. Una estrategia consiste en obligar a sus pesos a tomar  
#' valores pequeños, mediante la introducción de una penalización para los valores altos. Otra estrategia  
#' consiste eliminar neuronas durante el entrenamiento, con la idea de que la introducción de ruido en la   
#' salida de una capa puede hacer que la red ignore los patrones menos significativos.  
#'   
#' Para la regularización, vamos a utilizar la red de cuatro capas que era la que mayor sobreajuste  
#' presentaba y será la que mjor nos sirva para ilustrar la regularización.

#'  
#' Antes de empezar, vamos a entrenar otras 15 iteraciones a la red neuronal de cuatro capas, para  
#' para forzar aún más el sobreajuste y poder comparar mejor los efectos de la regularización

dense_network_4layers %>% fit(train_images, train_labels, epochs = 20, batch_size = 128, initial_epoch = 5)

#' Empezamos añadiendo regularización de la norma L1 de los pesos

dense_network_4layers_regL1 <- keras_model_sequential() %>% 
  layer_dense(units = 512, kernel_regularizer = regularizer_l1(0.001), activation = "relu", input_shape = c(28 * 28)) %>%
  layer_dense(units = 512, kernel_regularizer = regularizer_l1(0.001), activation = "relu") %>%
  layer_dense(units = 512, kernel_regularizer = regularizer_l1(0.001), activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

dense_network_4layers_regL1 %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

summary(dense_network_4layers_regL1)

#' Seguimos con regularización de la norma L2 de los pesos

dense_network_4layers_regL2 <- keras_model_sequential() %>% 
  layer_dense(units = 512, kernel_regularizer = regularizer_l2(0.001), activation = "relu", input_shape = c(28 * 28)) %>%
  layer_dense(units = 512, kernel_regularizer = regularizer_l2(0.001), activation = "relu") %>%
  layer_dense(units = 512, kernel_regularizer = regularizer_l2(0.001), activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

dense_network_4layers_regL2 %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

summary(dense_network_4layers_regL2)

#' Finalmente utilizamos un dropout del 50% de cada capa

dense_network_4layers_dropout <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax")

dense_network_4layers_dropout %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

summary(dense_network_4layers_dropout)

#' Se realiza el entrenamiento de las redes (utilizando 20 iteraciones)

dense_network_4layers_regL1 %>% fit(train_images, train_labels, epochs = 20, batch_size = 128)
dense_network_4layers_regL2 %>% fit(train_images, train_labels, epochs = 20, batch_size = 128)
dense_network_4layers_dropout %>% fit(train_images, train_labels, epochs = 20, batch_size = 128)

#' Se obtienen y se muestran los resultados

metrics_dn_4_layers_L1 <- dense_network_4layers_regL1 %>% evaluate(test_images, test_labels, verbose = 0)
metrics_dn_4_layers_L2 <- dense_network_4layers_regL2 %>% evaluate(test_images, test_labels, verbose = 0)
metrics_dn_4_layers_dropout <- dense_network_4layers_dropout %>% evaluate(test_images, test_labels, verbose = 0)
metrics_dn_4_layers_L1
metrics_dn_4_layers_L2
metrics_dn_4_layers_dropout

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
  nn4_layers = dense_network_4layers$metrics$val_loss,
  nn_4layers_regL1 = dense_network_4layers_regL1$metrics$val_loss
))

