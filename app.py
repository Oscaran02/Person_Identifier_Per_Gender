import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

data = pd.read_csv('./data/age_gender.csv')

dataframe = data.copy()
# Reduciendo los pixeles de la imagen a 48*48
dataframe["pixels"] = dataframe["pixels"].apply(lambda x: np.array(x.split(" "), dtype=np.float32).reshape(48, 48, 1))
# Imprimir tamaño de la imagen
print(dataframe["pixels"][0].shape)
# Ejemplo de los datos en el dataframe
print(dataframe.head())

# Imprimir las fotos con su edad y genero
fig, axes = plt.subplots(1, 4, figsize=(10, 5))
for i in range(4):
    random_face = np.random.choice(len(dataframe))
    axes[i].set_title('Edad: {0}, Genero: {1}'.format(dataframe['age'][random_face], dataframe['gender'][random_face]))
    axes[i].imshow(dataframe['pixels'][random_face])
    axes[i].axis('off')
plt.show()

# Esto ya que solo vamos a necesitar las imagenes y el genero para la validación
images = np.array(dataframe["pixels"].to_list())
genders = np.array(dataframe["gender"].to_list())

print("Shape of images: {}, Shape of genders: {}".format(images.shape, genders.shape))

images_train, image_test, genders_train, genders_test = train_test_split(images, genders, test_size=0.3,
                                                                         random_state=44)

# Data Generators
train_datagenerator = ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagenerator.flow(
    images_train, genders_train, batch_size=32
)

test_datagenerator = ImageDataGenerator(rescale=1 / 255)
test_generator = test_datagenerator.flow(
    image_test, genders_test, batch_size=32
)

# Callbacks
checkpoint_filepath = "./callback/ALLmodel_weights.h5"
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                             mode='max')
callbacks_list = [checkpoint]

earlystop = EarlyStopping(patience=6)
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc',
    patience=3,
    verbose=1,
)

callbacks = [earlystop, learning_rate_reduction, checkpoint]

# Model definition
inputs = Input(shape=(48, 48, 1))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dropout(0.5)(x)

outputs = Dense(1, activation='relu')(x)

model: Model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="Adam", loss="mean_squared_error", metrics=["accuracy"])

model.summary()

model.fit(
    train_generator,
    epochs=15,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.n // test_generator.batch_size,
)

results = pd.DataFrame(model.history.history)

# Imprimir los resultados de la perdida del modelo
results[['loss', 'val_loss']].plot(title='Model Loss')
plt.savefig('./loss.png')

# Imprimir los resultados de la precisión del modelo
results[['accuracy', 'val_accuracy']].plot(title='Precision del modelo')
plt.savefig('./accuracy.png')

# Mostrar el valor de accuracy y val_accuracy obtenidos
print("Accuracy: {}, Val_Accuracy: {}".format(results['accuracy'].iloc[-1], results['val_accuracy'].iloc[-1]))
# Mostrar el valor de loss y val_loss obtenidos
print("Loss: {}, Val_Loss: {}".format(results['loss'].iloc[-1], results['val_loss'].iloc[-1]))

# Matriz de confusión, para ver el resultado de la predicción frente a la realidad
class_names = ["Hombre", "Mujer"]
n = 10

image_batch, classes_batch = next(test_generator)

for batch in range(n):
    temp = next(test_generator)
    image_batch = np.concatenate((image_batch, temp[0]))
    classes_batch = np.concatenate((classes_batch, temp[1]))

classes_batch = classes_batch.tolist()
y_predict = model.predict(image_batch).reshape(32 * (n + 1))

y_temp = []
for x in y_predict:
    x = float(x)
    y_temp.append(0 if x < 0.5 else 1)
y_predict = y_temp

ConfusionMatrixDisplay.from_predictions(
    y_true=classes_batch,
    y_pred=y_predict,
    display_labels=class_names,
    cmap='Blues'
)
plt.savefig('./confusion_matrix.png')
plt.show()


# Imprimir el resultado de la predicción
def predict_one(model, num_images=None):
    image_batch, classes_batch = next(test_generator)
    predicted_batch = model.predict(image_batch)
    for k in range(0, image_batch.shape[0] if num_images is None else num_images):
        image = image_batch[k]
        real_class = class_names[0 if classes_batch[k] < 0.5 else 1]
        predicted_class = class_names[0 if predicted_batch[k] < 0.5 else 1]
        value_predicted = predicted_batch[k]
        isTrue = (real_class == predicted_class)
        plt.figure(k)
        plt.title(str("Predicción Correcta" if isTrue else "Predicción Incorrecta") + ' - Real: ' + real_class +
                  ' - ' + 'Predecid: ' + predicted_class + str(value_predicted))
        plt.axis('off')
        plt.savefig('./' + real_class + '_' + predicted_class + '_' + str(value_predicted) + '.png')
        plt.imshow(image)
        plt.show()


predict_one(model, num_images=10)
