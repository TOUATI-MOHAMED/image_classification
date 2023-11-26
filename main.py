import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array, load_img
#from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

#from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import load_model, Sequential, Model
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import Callback
import numpy as np
import os
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Flatten, Input
from tensorflow.keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt 
import seaborn as sns
import tensorflow as tf
from io import BytesIO
from pathlib import Path


# Création d'un callback pour mettre à jour l'interface utilisateur Streamlit
class StreamlitUpdateCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Mise à jour de l'interface utilisateur avec les informations de progression
        st.text(f"Epoch {epoch + 1}/{self.params['epochs']}")
        st.text(f"loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

# Décorateurs st.cache pour le chargement du modèle et l'entraînement
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def load_pretrained_model():
    # Chemin du fichier du modèle
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    x = base_model.layers[-5].output
    x = tf.keras.layers.Flatten()(x) 
    output = Dense(units=2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    for layer in model.layers[:-10]:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


@st.cache(suppress_st_warning=True)
def train_model(model, images_malade, labels_malade, images_normale, labels_normale):
    # Vérifier que les ensembles de données ne sont pas vides
    if len(images_malade) == 0 or len(labels_malade) == 0 or len(images_normale) == 0 or len(labels_normale) == 0:
        st.error("Erreur: Les ensembles de données pour au moins l'une des classes sont vides. Assurez-vous que les dossiers sélectionnés contiennent des images.")
    else:
        # Code d'entraînement ici
        #st.info("Début de l'entraînement du modèle...")

        # Concaténation des données
        X = np.concatenate([images_malade, images_normale], axis=0)
        y = np.concatenate([labels_malade, labels_normale], axis=0)
        # Convertir les étiquettes en format one-hot encoded
        y = to_categorical(y, num_classes=2)

        # Séparation des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Prétraitement des données pour ResNet50
        X_train = preprocess_input(X_train)
        X_test = preprocess_input(X_test)

        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[StreamlitUpdateCallback()])

        # Afficher les résultats en dehors de la fonction de cache
        display_results(model, history, X_test, y_test)

        return model, history

def display_results(model, history, X_test, y_test):
    # Évaluation du modèle
    y_pred = model.predict(X_test)
    y_pred_binary = np.round(y_pred)

    # Convertir les prédictions en classe unique
    y_pred_classes = np.argmax(y_pred_binary, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)


    # Affichage des résultats
    st.success("Entraînement terminé avec succès!")
    st.write(f"Précision du modèle : {accuracy}")

    # Création de la figure pour les graphiques
    plt.figure(figsize=(18, 6))

    # Courbe de la fonction de perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Courbe de la précision
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Affichage de la figure dans Streamlit
    st.pyplot(plt.gcf())

    # Affichage de la matrice de confusion en bas
    plt.figure(figsize=(12, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 14})
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Affichage de la deuxième figure dans Streamlit
    st.pyplot(plt.gcf())


def load_images(uploaded_files, label):
    images = []
    labels = []
    if uploaded_files is not None:
        for file in uploaded_files:
            img = Image.open(file)
            img = img.convert('RGB')  # Convert to RGB
            img = img.resize((224, 224))  # Resize the image
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = preprocess_input(img_array)  # Preprocess the image
            images.append(img_array)
            labels.append(label)

        # Stack images to create a batch
        images = np.vstack(images)

    return images, labels


# Titre de l'application
st.title("Classification d'images avec MobilNet")

# Chargement du modèle une seule fois
model = load_pretrained_model()

uploaded_malade = st.sidebar.file_uploader("Sélectionnez les images de la classe 0", type=["jpg", "jpeg","png"], accept_multiple_files=True)
first_label = st.sidebar.text_input("Nom de la classe 0", "first_label", key="first_label")

uploaded_normale = st.sidebar.file_uploader("Sélectionnez les images de la classe 1", type=["jpg", "jpeg","png"], accept_multiple_files=True)
second_label = st.sidebar.text_input("Nom de la classe 1", "second_label", key="second_label")

# Define the function to predict a new image
def predict_new_image(model, uploaded_image, first_label, second_label):
    if uploaded_image is not None:
        # Load and preprocess the image
        img = Image.open(uploaded_image)
        img = img.convert('RGB')  # Convert to RGB
        img = img.resize((224, 224))  # Resize the image
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
        img_array = preprocess_input(img_array)  # Preprocess the image

        # Predict the image
        prediction = model.predict(img_array)

        # Assuming binary classification
        predicted_class = int(np.round(prediction[0][0]))

        # Inverser la prédiction
        predicted_class = 1 if predicted_class == 0 else 0

        # Use the provided labels for display
        result_label = second_label if predicted_class == 1 else first_label

        probability = prediction[0][0] if predicted_class == 0 else 1 - prediction[0][0]

        st.write("Résultat de la prédiction : ", result_label)
        st.write(f"La probabilité de la prédiction est : {probability}")





# Define a session state to track the training status
if 'train_status' not in st.session_state:
    st.session_state.train_status = False

# Button to trigger model training
if st.sidebar.button("Entraînement"):
    if uploaded_malade is None or uploaded_normale is None:
        st.warning("Veuillez importer les fichiers d'images avant de lancer l'entraînement.")
    else:
        images_malade, first_label = load_images(uploaded_malade, 0)
        images_normale, second_label = load_images(uploaded_normale, 1)

        # Call the train_model function
        model, history = train_model(model, images_malade, first_label, images_normale, second_label)

        # Set the training status to True
        st.session_state.train_status = True


# Form to trigger image prediction (displayed only if training has been performed)
if st.session_state.train_status:      
    # Form to trigger image prediction
    with st.form("prediction_form"):
        uploaded_image = st.file_uploader("Sélectionnez une image pour la prédiction", type=["jpg", "jpeg","png"])
        # Display the uploaded image
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Image à prédire", use_column_width=True)
        submit_button = st.form_submit_button("Prédire")
        
    # Handle prediction after form submission
    if submit_button:
        predict_new_image(model, uploaded_image, first_label, second_label)


    # Button to download the trained model
    st.markdown("### Télécharger le modèle")
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        st.download_button(
            label="TensorFlow.js",
            data='my_model.tfjs',
            file_name='my_model.tfjs',
            mime='application/octet-stream',
        )

    with btn_col2:
        st.download_button(
            label="SavedModel",
            data='my_model.saved_model',
            file_name='my_model.saved_model',
            mime='application/octet-stream',
        )

    with btn_col3:
        st.download_button(
            label="HDF5",
            data='my_model.h5',
            file_name='my_model.h5',
            mime='application/octet-stream',
        )