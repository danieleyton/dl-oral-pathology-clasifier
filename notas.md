---
title: Notas para memoria de tesis
author: Daniel Leyton
---
# Buscar Data 
 + [dentistry iowa university](https://www.dentistry.uiowa.edu/oprm-atlas)
 + [Color photo atlas of dentistry](https://www.nycdentist.com/more/dental-photos/)
 + [DermNet NZ](https://www.dermnetnz.org/topics/mouth-problems/)
 + [Images and Media - UCSF Library](https://guides.ucsf.edu/c.php?g=100976&p=655198)
 + [Oral Disease Picture Gallery HardinMD UIOWA](https://web.archive.org/web/20170213170558/http://hardinmd.lib.uiowa.edu/dentpictures2.html)
 + [Stanford BioImage Search](https://lane.stanford.edu/search.html?q=Mouth&source=rl-images-all&auto=no&page=15)

# Investigar métodos CNN
 + LeNet
 + AlexNet
 + VGGNet
 + GoogLeNet
 + ResNet
 + ZFNet
 + Single Shot Multibox Detector (SSD) ?
 + Region-based CNN (R-CNN) ?
 + Faster R-CNN ?
 + Mask R-CNN ?
 + Fully Convolutional Neural Network (FCN) ?
 + ResNeXts
 + Xception
 + Inception

Separar imagenes en conjunto de pruebas, validacion.

Sintetizar entrada.

Avanzar en memoria, estado del arte, modelos que hayan funcionado bien, cercano al problema (uno o dos parrafos). (revisar y ayudarse en paperswithcode.com)

Entrenar la red convolucional con los pocos datos que tenemos va a resultar en un sobreajuste, la técnica de Data Augmentation nos puede ayudar a resolver este problema y Keras tiene herramientas que nos pueden ayudar para esto [Link](https://towardsdatascience.com/classify-butterfly-images-with-deep-learning-in-keras-b3101fe0f98).

[Keras image preprocessing](https://keras.io/api/preprocessing/image/)

paperswithcode.com browse state of the art, entrega los papers con el código (duh..)

En el paper Detection and diagnosis pf dental caries using a deep learning-based convolutional neural network algorithm usaron una red GoogLeNet Inception v3 CNN pre-entrenada para el preprocesamiento, y los datasets fueron entrenados usando transfer learning. Un total de 9 modulos de inception fueron usados, incluyendo un clasificador auxiliar, dos capas totalmente conectadas, y funciones de softmax. "The training set was separated randomly into 32 batches for every epoch, and 1000 epochs were run at a learning rate of 0.01. To provide better detection of dental caries, fine tuning was used to optimize the weights and improve the output power by adjusting the hyperparameters"

Buscar y leer el paper Deep Learning for Automated Detection of Cyst and Tumors of the Jaw in Panoramic Radiographs
