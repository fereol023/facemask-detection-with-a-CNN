# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 09:54:54 2022

@author: gbeno
"""
import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from  matplotlib import pyplot as plt


from tensorflow import keras
from keras.layers import Dense
from keras.layers import Dropout # 
from keras.layers import Flatten # couche qui applatit les inputs
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import SGD
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model


from keras.models import Sequential
from skimage.transform import resize

from scipy import misc




def load_data(repertoire, img_width, img_height) :
    '''
    Fonction d'importation des images au format img_width*img_height 
    spécifié depuis le répertoire 
    '''
    label = os.listdir(repertoire)
    
    dataset = []
    
    for image_label in label:
        images = os.listdir(repertoire+"/"+image_label)
       
        for image in images:
            img = mpimg.imread(repertoire+"/"+image_label+"/"+image)
            # normalisation de toutes les images 
            img = resize(img, (img_width, img_height))
            dataset.append((img,image_label))
    X = []
    Y = []
    
    for input_,image_label in dataset:
        X.append(input_)
        Y.append(label.index(image_label))
    
    X = np.array(X)
    Y = np.array(Y)
    
    X_train,y_train, = X,Y
    
    data_set = (X_train,y_train)
    
    #print(np.shape(data_set[0]))
    #print(type(data_set))
    #print(type(data_set[0]))
    #print(type(data_set[1]))

    return data_set
    

def modele(data_set, epochs, lrate, loss_fun = "categorical_crossentropy", seed=2309) :
        
    # on fixe le "seed" à 2309 pour obtenir les mêmes réseultats à chaque exécution
    np.random.seed(seed)
    
    # on charge les inputs 
    (X,y) = data_set
    
    # on encode y en variable categorielle binaire
    y = np_utils.to_categorical(y)
    
    # on récupère le nombre de catégories de y : 2
    num_classes = y.shape[1]
    
    # on créé le modèle à partir de l'API sequential
    # le modèle est une superposition des couches prises dans Keras
    # les couches sont superposées dans l'ordre de leur ajout au modèle
    model = Sequential()
        
        # première couche de convolution :  
        # extrait les caractéristiques de l'image
        #  input_shape : format des données
    model.add(Conv2D(32, (3, 3), input_shape = (100, 100, 3), padding = 'same', 
                     activation = 'relu', kernel_constraint = maxnorm(3)))
        
        # dropout désactive aléatoirement 20% des informations transmises par la  
        # couche précédente pour éviter le surapprentissage
    model.add(Dropout(0.2))
    
        # deuxieme couche de convolution 
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', 
              kernel_constraint = maxnorm(3)))

    model.add(MaxPooling2D(pool_size = (2, 2)))
        
    model.add(Flatten())
        
    model.add(Dense(800, activation = 'relu', kernel_constraint = maxnorm(3)))
    
        # désactive aléatoirement 50% des informations fournies par la couche précédente
    model.add(Dropout(rate = 0.5)) 
        
    model.add(Dense(num_classes, activation = 'softmax'))
    
    
    # Compile model
    decay = lrate/epochs
    
    sgd = SGD(lr = lrate, momentum = 0.9, decay = decay, nesterov = False)
    
    model.compile(loss = loss_fun, optimizer = sgd, metrics = ['accuracy'])
    
    print(model.summary())
    
    """
    # Callbacks
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss', min_delta = 0, patience = 0, verbose = 0, mode = 'auto')]
   
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs', 
        histogram_freq = 0, batch_size = 32, write_graph = True, write_grads = False, 
        write_images = True, embeddings_freq = 0, embeddings_layer_names = None, 
        embeddings_metadata = None)]
    """
    return model
    

def learning(model, data_set, epochs, seed=2309) :
    '''
    Implémentation 
    '''
     # fix random seed at 2309 for reproducibility
    np.random.seed(seed)
    
    # inputs : load_data 
    (X_train,y_train) = data_set
    
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    
    # Fit the model
    model.fit(X_train, y_train, epochs) 
      
def evaluation(model, data_set) :
    # preparation
    (X, y) = data_set
    y = np_utils.to_categorical(y)
    
    # evaluation du model sur les données
    scores = model.evaluate(X, y, verbose = 0)
    print("Accuracy : %.2f%%" % (scores[1]*100))
    return scores

def testing(new_data, model) :
    print("Evaluation sur les données test ---------------------")
    resultats = evaluation(new_data, model)
    print("Test loss, Test accuracy:", resultats)
    print("Fin testing")
    
def prediction(new_data, model) :
    # preparation
    (X_test,y_test) = new_data
    # on new data using `predict`
    print("Predictions ---------------------")
    predictions = model.predict(X_test[:1])
    print(predictions)
    print("Predictions shape:", predictions.shape)

def sauvegarde(nom='Facemask_detection_modele') :
    # sauvegarde
    modele.save(str(nom))
    print("Modèle sauvegardé dans le répertoire : "+str(nom))
    
def rechargement(nom='Facemask_detection_modele') :
    # rechargement prend en paramètre le nom du répertoire du modèle
    modele = keras.models.load_model(str(nom))
    print("Modèle "+str(nom)+" chargé !")
    return modele

# Fonction principale
def face_mask_detection(Dtrain, Dtest, epochs, lrate, loss_fun) :
    # apprentissage
    choix_train = str(input("Apprendre(1) ou Restaurer(2) : \n"))
    if choix_train == "1" :
        model = modele(Dtrain, epochs, lrate, loss_fun)
        learning(model, Dtrain, epochs)
        evaluation(model, Dtrain)
        #sauvegarde()
    else :
        model = rechargement()
    
    choix_test = str(input("Tester(1) ou pas(2) : \n"))
    # test
    if choix_test == "1" :
        testing(Dtest, model)
        prediction(Dtest, model)
    else :
        pass
    
    print("Fin du programme !")


if __name__ == '__main__':
    
    # ramène sur le répertoire du projet
    repertoire_courant = os.getcwd()
    os.chdir(repertoire_courant)
    
    # format image pour normalisation 
    img_width, img_height = 100, 100
    
    # paramètres du modèle
    epochs = 2  # nombre de tours complet sur Dtrain pour l'apprentissage
    lrate = 0.01 # learning rate
    loss_fun = "categorical_crossentropy" # fonction de perte 
    
    # chemin vers data train et vers data test
    dtrain_dir = "facemask_dataset/Train"
    dtest_dir = "facemask_dataset/newdata"
    
    # chargement des données 
    Dtrain = load_data(dtrain_dir, img_width, img_height)
    print("Données d'apprentissage chargées !")
    Dtest = load_data(dtest_dir, img_width, img_height)
    print("Données du test chargées !")
    
    # exécution de la fonction principale
    face_mask_detection(Dtrain, Dtest, epochs, lrate, loss_fun)
    
    