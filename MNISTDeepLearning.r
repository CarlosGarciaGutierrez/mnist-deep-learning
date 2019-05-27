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
library(ggplot2)
library(tidyr)

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

#' Se reordenan los datos para poder ser usados como entrada de las redes densas y se escalan los valores  
#' RGB de las imágenes para que estén en el intervalo [0, 1], asimismo se transforman las etiquetas a     
#' valores binarios, según el dígito que representen

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
  
metrics_dn_2layers <- dense_network_2layers %>% evaluate(test_images, test_labels)
metrics_dn_3layers <- dense_network_3layers %>% evaluate(test_images, test_labels)
metrics_dn_4layers <- dense_network_4layers %>% evaluate(test_images, test_labels)
metrics_dn_2layers
metrics_dn_3layers
metrics_dn_4layers

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

#' Finalmente utilizamos un dropout (del 50% de cada capa)

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

#' Antes de continuar, vamos a entrenar otras 5 iteraciones a la red neuronal de cuatro capas, para  
#' para forzar aún más el sobreajuste y poder comparar mejor los efectos de la regularización

dense_network_4layers_hist <- dense_network_4layers %>% 
  fit(train_images, train_labels, epochs = 10, batch_size = 128, initial_epoch = 5)

#' Se realiza el entrenamiento de las redes (utilizando 10 iteraciones)

dense_network_4layers_regL1_hist <- dense_network_4layers_regL1 %>%
  fit(train_images, train_labels, epochs = 10, batch_size = 128)
dense_network_4layers_regL2_hist <- dense_network_4layers_regL2 %>%
  fit(train_images, train_labels, epochs = 10, batch_size = 128)
dense_network_4layers_dropout_hist <- dense_network_4layers_dropout %>%
  fit(train_images, train_labels, epochs = 10, batch_size = 128)

#' Se obtienen y se muestran los resultados

metrics_dn_4layers <- dense_network_4layers %>% evaluate(test_images, test_labels)
metrics_dn_4layers_L1 <- dense_network_4layers_regL1 %>% evaluate(test_images, test_labels)
metrics_dn_4layers_L2 <- dense_network_4layers_regL2 %>% evaluate(test_images, test_labels)
metrics_dn_4layers_dropout <- dense_network_4layers_dropout %>% evaluate(test_images, test_labels)
metrics_dn_4layers
metrics_dn_4layers_L1
metrics_dn_4layers_L2
metrics_dn_4layers_dropout

#' Como se puede observar en los resultados el sobreajuste únicamente mejora con la regularización que utiliza  
#' "dropout", pero no con las de las normas de los pesos. Una interpretación plausible podría ser que los pesos  
#' de la red ya eran pequeños antes de introducir la regularización y que, por otro lado, existe ruido en los  
#' los datos de origen y, gracias al "dropout", se evita que el modelo sobreajuste al no tener en cuenta dicho   
#' ruido.  
#'   
#' **NOTA**: lo ideal hubisese sido mostrar una gráfica con la evulucion de la función de pérdida a lo largo  
#' de las iteraciones, para cada uno de los cuatros modelos, en vez de mostrar en texto los resultados finales  
#' de la última iteración. Sin embargo, no me ha sido posible mostar ningún gráfico en mi ordenador, ya que cada  
#' vez que se intentaba lanzar uno RStudio empezaba a consumir memoria y tras un par de minutos acababa quedándose  
#' bloqueado, haciendo necesario finalizar el proceso manualmente. No he sido capaz de solucionar dicho problema.

 
#'  
#' **Redes convolucionales**

#'  
#' Empezamos creando las redes convolucionales. Debemos asegurarnos de que reciba como entrada tensores de  
#' tamaño 28x28x1 y de que la salida tenga 10 neuronas (una por cada categoría/dígito). El tamaño del  
#' "pooling", de los "kernels" y el número de filtros varían entre cada uno de los ejemplos.

conv_network_a <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(2, 2), activation = "relu", input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 32, kernel_size = c(2, 2), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 32, kernel_size = c(2, 2), activation = "relu") %>%
  layer_flatten() %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

conv_network_a %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

summary(conv_network_a)

conv_network_b <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(2, 2), activation = "relu", input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(3, 3)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(2, 2), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(3, 3)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(2, 2), activation = "relu") %>%
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

conv_network_b %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

summary(conv_network_b)

conv_network_c <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(4, 4), activation = "relu", input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(4, 4), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(4, 4), activation = "relu") %>%
  layer_flatten() %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

conv_network_c %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

summary(conv_network_c)

#' Obtenemos de nuevo las imágenes/etiquetas de entrenamiento y de test
train_images <- mnist$train$x[sample_array,,]
test_images <- mnist$test$x

#' Se reordenan los datos para poder ser usados como entrada de las redes convolucionales y se escalan los 
#' valores RGB de las imágenes para que estén en el intervalo [0, 1]

train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))
test_images <- test_images / 255

#' Se realiza el entrenamiento de las redes

conv_network_a %>% fit(train_images, train_labels, epochs = 5, batch_size = 64)
conv_network_b %>% fit(train_images, train_labels, epochs = 5, batch_size = 64)
conv_network_c %>% fit(train_images, train_labels, epochs = 5, batch_size = 64)

#' Se obtienen y se muestran los resultados

metrics_cn_a <- conv_network_a %>% evaluate(test_images, test_labels)
metrics_cn_b <- conv_network_b %>% evaluate(test_images, test_labels)
metrics_cn_c <- conv_network_c %>% evaluate(test_images, test_labels)

metrics_cn_a
metrics_cn_b
metrics_cn_c

#' Se puede observar como resultado que los peores resultados se obtienen en red convolucional B. Esto era  
#' esperable ya que la combinación del tamaño de kernel y de operadores de pooling no es la adecuada. Los  
#' mejores resultados se obtienen con la red convolucional C, probablemente debiso a su número de filtros,  
#' pero a costa de un notabilísimo coste computacional.

#'  
#' Partiremos ahora la red convolucional A para realizar las combinaciones de capas de de convolución y de  
#' pooling que se solicitan. Podría partirse también de la red convolucional C, y muy probablemente se  
#' obtendrían mejores resultados, pero el coste computacional sería elevadísimo. La red convolucional A  
#' es una elección más equilibrada.

#'  
#' Primera combinación: capa convolucional transpuesta 2D y capa de max_pooling 1D  

#conv_network_a1 <- keras_model_sequential() %>% 
#  layer_conv_2d_transpose(filters = 32, kernel_size = c(2, 2), activation = "relu", input_shape = c(28, 28, 1)) %>% 
#  layer_max_pooling_1d(pool_size = 2) %>% 
#  layer_conv_2d_transpose(filters = 32, kernel_size = c(2, 2), activation = "relu") %>% 
#  layer_max_pooling_1d(pool_size = 2) %>% 
#  layer_conv_2d_transpose(filters = 32, kernel_size = c(2, 2), activation = "relu") %>%
#  layer_flatten() %>% 
#  layer_dense(units = 32, activation = "relu") %>% 
#  layer_dense(units = 10, activation = "softmax")

#'  
#' El resultado es que no se puede realizar pooling 1D sobre el resultado de una capaz de mayor dimensión  

#'  
#' Segunda combinación: capa convolucional transpuesta 2D y capa de max_pooling 2D

conv_network_a2 <- keras_model_sequential() %>% 
  layer_conv_2d_transpose(filters = 32, kernel_size = c(2, 2), activation = "relu", input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d_transpose(filters = 32, kernel_size = c(2, 2), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d_transpose(filters = 32, kernel_size = c(2, 2), activation = "relu") %>%
  layer_flatten() %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

conv_network_a2 %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

summary(conv_network_a2)

#' El resultado del entrenamiento y la evaluación se muestra a contionuación

conv_network_a2 %>% fit(train_images, train_labels, epochs = 5, batch_size = 64)
metrics_cn_a2 <- conv_network_c %>% evaluate(test_images, test_labels)
metrics_cn_a2

#'  
#' Tercera combinación: capa convolucional transpuesta 2D y capa de average_pooling 2D

conv_network_a3 <- keras_model_sequential() %>% 
  layer_conv_2d_transpose(filters = 32, kernel_size = c(2, 2), activation = "relu", input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d_transpose(filters = 32, kernel_size = c(2, 2), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d_transpose(filters = 32, kernel_size = c(2, 2), activation = "relu") %>%
  layer_flatten() %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

conv_network_a3 %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

summary(conv_network_a3)

#' El resultado se muestra a continuación del entrenamiento y la evaluación se muestra a contionuación

conv_network_a3 %>% fit(train_images, train_labels, epochs = 5, batch_size = 64)
metrics_cn_a3 <- conv_network_c %>% evaluate(test_images, test_labels)
metrics_cn_a3

#'  
#' Cuarta combinación: capa convolucional 3D y capa de max_pooling 2D

#conv_network_a4 <- keras_model_sequential() %>% 
#  layer_conv_3d(filters = 32, kernel_size = c(2, 2, 2), activation = "relu", input_shape = c(28, 28, 1)) %>% 
#  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
#  layer_conv_3d_transpose(filters = 32, kernel_size = c(2, 2, 2), activation = "relu") %>% 
#  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
#  layer_conv_3d_transpose(filters = 32, kernel_size = c(2, 2, 2), activation = "relu") %>%
#  layer_flatten() %>% 
#  layer_dense(units = 32, activation = "relu") %>% 
#  layer_dense(units = 10, activation = "softmax")

#'  
#' El resultado es que la capa de entrada no puede ser de dimensión mayor que la dimensión de los datos  

#'  
#' Quinta combinación: capa convolucional 2D y capa de global_max_pooling 2D

#conv_network_a5 <- keras_model_sequential() %>% 
#  layer_conv_2d(filters = 32, kernel_size = c(2, 2), activation = "relu", input_shape = c(28, 28, 1)) %>% 
#  layer_global_max_pooling_2d() %>% 
#  layer_conv_2d(filters = 32, kernel_size = c(2, 2), activation = "relu") %>% 
#  layer_global_max_pooling_2d() %>% 
#  layer_conv_2d(filters = 32, kernel_size = c(2, 2), activation = "relu") %>%
#  layer_flatten() %>% 
#  layer_dense(units = 32, activation = "relu") %>% 
#  layer_dense(units = 10, activation = "softmax")

#'  
#' El resultado es que la capa pooling global devuelve una salida de dimensión menor a la necesaria por la  
#' siguiente capa convolucional  
