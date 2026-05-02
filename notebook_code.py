import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import joblib

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

import warnings
warnings.filterwarnings('ignore')

print('Tensorflow version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('Using cuda devices (GPU)' if gpus else 'GPU not available, using CPU')

# Path and parameters
DATADIR = '/kaggle/input/cat-dog-images-for-classification/cat_dog'
H, W = 224, 224
EPOCHS = 50
BATCH_SIZE = 32
SEED = 42

df = pd.read_csv('/kaggle/input/cat-dog-images-for-classification/cat_dog.csv')
df['filenames'] = df['image']
df['labels'] = df['labels'].map(lambda x: 'dog' if x == 1 else 'cat')
df.drop(columns=['image'], inplace=True)
df.head()

df.labels.value_counts()

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df['labels'])
valid_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=SEED, stratify=temp_df['labels'])
train_df.sample(frac=1, random_state=SEED)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print('Train size:', len(train_df))
print('Validation size:', len(valid_df))
print('Test size:', len(test_df))

train_df.to_csv('train.csv')
valid_df.to_csv('valid.csv')
test_df.to_csv('test.csv')

dogs = list(df[df.labels == 'dog'].filenames)
cats = list(df[df.labels == 'cat'].filenames)

def get_side(img, side_type, n=5):
    h, w, c = img.shape
    if side_type == 'horizontal':
        return np.ones((h, n, c))
    return np.ones((n, w, c))

def show_gallery(data, n, title):
    images = []
    vertical_images = []

    for i in range(n * n):
        img = load_img(os.path.join(DATADIR, data[i]), target_size=(W, H))
        img = img_to_array(img)
        hside = get_side(img, side_type='horizontal')
        images.append(img)
        images.append(hside)

        if (i + 1) % n == 0:
            himage = np.hstack((images))
            vside = get_side(himage, side_type="vertical")
            vertical_images.append(himage)
            vertical_images.append(vside)
            
            images = []

    gallery = np.vstack((vertical_images))
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.imshow(gallery.astype(np.uint8))
    plt.savefig(f'gallery_{title}.jpg', dpi=300)
    plt.show()

show_gallery(dogs, n=10, title='dog')

show_gallery(cats, n=10, title='cat')

@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy > 0, 'float32') * tf.cast(x > 0, 'float32') * dy
    return tf.nn.relu(x), grad

class GuidedBackprop:
    def __init__(self,model, layerName=None):
        self.model = model
        self.layerName = layerName
        self.gbModel = self.build_guided_model()
        
        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

    def build_guided_model(self):
        gbModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output]
        )
        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer, "activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
        
        return gbModel
    
    def guided_backprop(self, images, upsample_size):
        """Guided Backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            outputs = self.gbModel(inputs)

        grads = tape.gradient(outputs, inputs)[0]

        saliency = cv2.resize(np.asarray(grads), upsample_size)

        return saliency

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

class GradCAM:
    def __init__(self, model, layerName=None):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        self.layerName = layerName
            
        if self.layerName == None:
            self.layerName = self.find_target_layer()
    
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")
            
    def compute_heatmap(self, image, classIdx, upsample_size, eps=1e-5):
        gradModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output, self.model.outputs]
        )
        # record operations for automatic differentiation
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs) # preds after softmax
            loss = preds[0][:, classIdx]
        
        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))
        
        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0,1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)
        
        # Apply reLU
        cam = np.maximum(cam, 0)
        cam = cam/np.max(cam)
        cam = cv2.resize(cam, upsample_size,interpolation=cv2.INTER_LINEAR)
        
        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1,1,3])
        
        return cam3
    
def overlay_gradCAM(img, cam3):
    cam3 = np.uint8(255 * cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
    
    new_img = 0.3 * cam3 + 0.5 * img
    
    return (new_img * 255.0 / new_img.max()).astype("uint8")

def show_gradCAMs(model, gradCAM, GuidedBP, im_ls, n, classes):
    """
    model: softmax layer
    """

    plt.subplots(figsize=(30, 10 * n))
    k = 1
    
    for i in range(n):
        img = cv2.imread(os.path.join(DATADIR, im_ls[i]))
        upsample_size = (img.shape[1], img.shape[0])

        # Show original image
        plt.subplot(n, 3, k)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Filename: {im_ls[i]}', fontsize=20)
        plt.axis('off')

        # Show overlayed Grad
        plt.subplot(n, 3, k + 1)
        im = img_to_array(load_img(os.path.join(DATADIR, im_ls[i]), target_size=(W, H)))
        x = np.expand_dims(im, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        idx = preds.argmax()

        cam3 = gradCAM.compute_heatmap(image=x, classIdx=idx, upsample_size=upsample_size)
        new_img = overlay_gradCAM(img, cam3)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        plt.imshow(new_img)
        plt.title(f'GradCAM - Pred: {classes[idx]}. Prob: {preds.max()}', fontsize=20)
        plt.axis('off')

        # Show Guided GradCAM
        plt.subplot(n, 3, k + 2)
        gb = GuidedBP.guided_backprop(x, upsample_size)
        guided_gradcam = deprocess_image(gb * cam3)
        guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
        plt.imshow(guided_gradcam)
        plt.title("Guided GradCAM", fontsize=20)
        plt.axis("off")
        
        k += 3
        
    plt.show()

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    DATADIR,
    x_col='filenames',
    y_col='labels',
    target_size=(W, H),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_generator = valid_datagen.flow_from_dataframe(
    valid_df,
    DATADIR,
    x_col='filenames',
    y_col='labels',
    target_size=(W, H),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    DATADIR,
    x_col='filenames',
    y_col='labels',
    target_size=(W, H),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# resnet = ResNet50V2(weights='imagenet', pooling='avg', include_top=False)
# for layer in resnet.layers:
#     layer.trainable = False

# fc1 = Dense(128)(resnet.layers[-1].output)
# fc2 = Dense(128, name='dense_feature')(fc1)
# dropout = Dropout(0.5)(fc2)
# outputs = Dense(2, activation='softmax')(dropout)

# model = Model(inputs=resnet.input, outputs=outputs)

# sgd = optimizers.SGD(learning_rate=1e-3, weight_decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# model.summary()

# history = model.fit(
#     train_generator,
#     validation_data=valid_generator,
#     epochs=10
# )

# model.save('resnet50v2.h5')

# plt.figure(figsize=(12, 4))

# plt.subplot(121)
# plt.plot(history.history['loss'], label='Train') 
# plt.plot(history.history['val_loss'], label='Validation')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(122)
# plt.plot(history.history['accuracy'], label='Train')
# plt.plot(history.history['val_accuracy'], label='Validtion')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.savefig('history.png')

# plt.show()

model = load_model('/kaggle/input/models/tensorflow2/default/2/resnet50v2.h5')

test_loss, test_accuracy = model.evaluate(test_generator)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

model_logit = Model(model.input, model.layers[-1].output)
fctrained_gradCAM = GradCAM(model_logit, layerName='conv5_block3_out')
fctrained_guidedBP = GuidedBackprop(model, layerName='conv5_block3_out')

y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

results = test_df.copy()
results['predict'] = y_pred
true_dogs = list(results[(results.labels == 'dog') & (results.predict == 1)].filenames)
true_cats = list(results[(results.labels == 'cat') & (results.predict == 0)].filenames)
wrong_class = [x for x in results.filenames if x not in (true_dogs + true_cats)]

len(wrong_class)

show_gradCAMs(model, fctrained_gradCAM, fctrained_guidedBP, true_dogs, n=5, classes={0: 'cat', 1: 'dog'})

show_gradCAMs(model, fctrained_gradCAM, fctrained_guidedBP, true_cats, n=5, classes={0: 'cat', 1: 'dog'})

show_gradCAMs(model, fctrained_gradCAM, fctrained_guidedBP, wrong_class, n=5, classes={0: 'cat', 1: 'dog'})

class FeatureExtractor:
    def __init__(self, model, layerName=None):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        self.layerName = layerName
            
        if self.layerName == None:
            self.layerName = self.find_target_layer()
    
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")
            
    def compute_c_hp(self, image, classIdx):
        gradModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output, self.model.outputs]
        )
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs) 
            loss = preds[0][:, classIdx]
        
        grads = tape.gradient(loss, convOuts)
        if grads is None:
            raise ValueError("Gradients is None")
        
        c_hp = (grads * convOuts)[0]
        
        c_hp_flat = tf.reshape(c_hp, [-1])
        c_hp_flat = tf.math.l2_normalize(c_hp_flat)
        return c_hp_flat.numpy()

def get_chp_on_dataframe(extractor, df):
    features = []
    labels = []
    for path in tqdm(df.filenames):

        img = img_to_array(load_img(os.path.join(DATADIR, path), target_size=(W, H)))
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x, verbose=0)
        idx = preds.argmax()
        
        c_hp = extractor.compute_c_hp(image=x, classIdx=idx)
        features.append(c_hp)
        labels.append('dog' if idx == 1 else 'cat')

    return np.array(features), np.array(labels)

extractor = FeatureExtractor(model, layerName='dense_feature')

(X_train, y_train), y_train_true = get_chp_on_dataframe(extractor, train_df), train_df.labels
(X_test, y_test), y_test_true = get_chp_on_dataframe(extractor, test_df), test_df.labels

X_train.shape, y_train.shape, X_test.shape, y_test.shape

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('Accuracy:', accuracy_score(y_test_true, y_pred)) # Accuracy with true labels
print('Agreement:', accuracy_score(y_test, y_pred)) # Fidelity

joblib.dump(knn, 'knn_chp_norm_model.pkl')

# knn = joblib.load('/kaggle/input/models/tensorflow2/default/2/knn_chp_model.pkl')

rand_idx = np.random.randint(0, len(test_df))
path = test_df.filenames[rand_idx]
# path = 'dog.6131.jpg'
classes = {0: 'cat', 1: 'dog'}

img = cv2.imread(os.path.join(DATADIR, path))
upsample_size = (img.shape[1], img.shape[0])

plt.subplots(figsize=(30, 20))

# Show original image
plt.subplot(231)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f'Filename: {path}', fontsize=20)
plt.axis('off')

# Show overlayed Grad
plt.subplot(232)
im = img_to_array(load_img(os.path.join(DATADIR, path), target_size=(W, H)))
x = np.expand_dims(im, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
idx = preds.argmax()

cam3 = fctrained_gradCAM.compute_heatmap(image=x, classIdx=idx, upsample_size=upsample_size)
new_img = overlay_gradCAM(img, cam3)
new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
plt.imshow(new_img)
plt.title(f'GradCAM - Pred: {classes[idx]}. Prob: {preds.max()}', fontsize=20)
plt.axis('off')

# Show Guided GradCAM
plt.subplot(233)
gb = fctrained_guidedBP.guided_backprop(x, upsample_size)
guided_gradcam = deprocess_image(gb * cam3)
guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
plt.imshow(guided_gradcam)
plt.title("Guided GradCAM", fontsize=20)
plt.axis("off")

######################### Retrieved ##################################

c_hp = extractor.compute_c_hp(image=x, classIdx=idx)
knn_index = knn.kneighbors(c_hp.reshape(1, -1), n_neighbors=1, return_distance=False)[0, 0]
c_hp_retrieved = knn._fit_X[knn_index]
retrieved_path = train_df.filenames[knn_index]
retrieved_label = train_df.labels[knn_index]

img = cv2.imread(os.path.join(DATADIR, retrieved_path))
upsample_size = (img.shape[1], img.shape[0])

# Show retrieved image
plt.subplot(234)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f'Retrieved Filename: {retrieved_path}', fontsize=20)
plt.axis('off')

# Show overlayed Grad
plt.subplot(235)
im = img_to_array(load_img(os.path.join(DATADIR, retrieved_path), target_size=(W, H)))
x = np.expand_dims(im, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
idx = preds.argmax()

cam3 = fctrained_gradCAM.compute_heatmap(image=x, classIdx=idx, upsample_size=upsample_size)
new_img = overlay_gradCAM(img, cam3)
new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
plt.imshow(new_img)
plt.title(f'GradCAM - Pred: {classes[idx]}. Prob: {preds.max()}', fontsize=20)
plt.axis('off')

# Show Guided GradCAM
plt.subplot(236)
gb = fctrained_guidedBP.guided_backprop(x, upsample_size)
guided_gradcam = deprocess_image(gb * cam3)
guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
plt.imshow(guided_gradcam)
plt.title("Guided GradCAM", fontsize=20)
plt.axis("off")

plt.show()

from sklearn.metrics.pairwise import cosine_similarity

np.round(cosine_similarity(np.array(c_hp).reshape(1, -1), 
                           np.array(c_hp_retrieved).reshape(1, -1)), 2)

cosine_similarities = []
euclidean_distances = []

for path in tqdm(test_df.filenames):
    img = cv2.imread(os.path.join(DATADIR, path))
    upsample_size = (img.shape[1], img.shape[0])
    
    im = img_to_array(load_img(os.path.join(DATADIR, path), target_size=(W, H)))
    x = np.expand_dims(im, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x, verbose=0)
    idx = preds.argmax()
    
    ######################### Retrieved ##################################
    
    c_hp = extractor.compute_c_hp(image=x, classIdx=idx)
    L2_distance, knn_index = knn.kneighbors(c_hp.reshape(1, -1), n_neighbors=1, return_distance=True)
    c_hp_retrieved = knn._fit_X[knn_index[0, 0]]

    cosine = np.round(cosine_similarity(np.array(c_hp).reshape(1, -1), 
                                       np.array(c_hp_retrieved).reshape(1, -1)), 2)[0, 0]
    cosine_similarities.append(cosine)
    euclidean_distances.append(L2_distance[0, 0])

print('Mean Cosine Similarity:', np.mean(np.round(cosine_similarities, 2)))
print('Mean L2:', np.mean(np.round(euclidean_distances, 2)))

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.title('Phân phối Cosine Similarity')
plt.hist(np.round(cosine_similarities, 2), bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Cosine Similarity')
plt.ylabel('Số lượng mẫu')

plt.subplot(122)
plt.title('Phân phối Euclidean Distance')
plt.hist(np.round(euclidean_distances, 2), bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Euclidean Distance')
plt.ylabel('Số lượng mẫu')

plt.tight_layout()
plt.show()

prototype_dog = knn._fit_X[knn._y == 1].mean(axis=0)
prototype_cat = knn._fit_X[knn._y == 0].mean(axis=0)

from sklearn.decomposition import PCA

features_with_prototypes = np.vstack([knn._fit_X, prototype_cat, prototype_dog])

# 3. Giảm chiều
pca = PCA(n_components=2)
reduced = pca.fit_transform(features_with_prototypes)

reduced_features = reduced[:-2]
reduced_prototypes = reduced[-2:]

plt.figure(figsize=(8,6))
plt.scatter(reduced_features[knn._y==0,0], reduced_features[knn._y==0,1], label='Cat', alpha=0.6)
plt.scatter(reduced_features[knn._y==1,0], reduced_features[knn._y==1,1], label='Dog', alpha=0.6)
plt.scatter(reduced_prototypes[0,0], reduced_prototypes[0,1], marker='X', s=200, color='blue', label='Prototype Cat')
plt.scatter(reduced_prototypes[1,0], reduced_prototypes[1,1], marker='X', s=200, color='red', label='Prototype Dog')

plt.legend()
plt.title('Visualization of Features and Prototypes')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()

feature_norm = []
for x in tqdm(knn._fit_X):
    feature_norm.append(tf.math.l2_normalize(x))
feature_norm = np.array(feature_norm)
feature_norm.shape

prototype_dog = feature_norm[knn._y == 1].mean(axis=0)
prototype_cat = feature_norm[knn._y == 0].mean(axis=0)

from sklearn.decomposition import PCA

features_with_prototypes = np.vstack([feature_norm, prototype_cat, prototype_dog])

# 3. Giảm chiều
pca = PCA(n_components=2)
reduced = pca.fit_transform(features_with_prototypes)

reduced_features = reduced[:-2]
reduced_prototypes = reduced[-2:]

plt.figure(figsize=(8,6))
plt.scatter(reduced_features[knn._y==0,0], reduced_features[knn._y==0,1], label='Cat', alpha=0.6)
plt.scatter(reduced_features[knn._y==1,0], reduced_features[knn._y==1,1], label='Dog', alpha=0.6)
plt.scatter(reduced_prototypes[0,0], reduced_prototypes[0,1], marker='X', s=200, color='blue', label='Prototype Cat')
plt.scatter(reduced_prototypes[1,0], reduced_prototypes[1,1], marker='X', s=200, color='red', label='Prototype Dog')

plt.legend()
plt.title('Visualization of Features and Prototypes')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()

import plotly.express as px

pca = PCA(n_components=3)
X_pca = pca.fit_transform(knn._fit_X)

fig = px.scatter_3d(
    x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
    color=['Cat' if label == 0 else 'Dog' for label in knn._y],
    labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
    title='PCA 3D Visualization (Plotly)',
    opacity=0.7
)

fig.update_layout(scene=dict(
    xaxis_title='PC1',
    yaxis_title='PC2',
    zaxis_title='PC3'
))
fig.show()

import plotly.express as px

pca = PCA(n_components=3)
X_pca = pca.fit_transform(feature_norm)

fig = px.scatter_3d(
    x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
    color=['Cat' if label == 0 else 'Dog' for label in knn._y],
    labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
    title='PCA 3D Visualization (Plotly)',
    opacity=0.7
)

fig.update_layout(scene=dict(
    xaxis_title='PC1',
    yaxis_title='PC2',
    zaxis_title='PC3'
))
fig.show()

from sklearn.manifold import TSNE

# Reduce to 3D
tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(knn._fit_X)

# Plot with Plotly
fig = px.scatter_3d(
    x=X_tsne[:, 0], y=X_tsne[:, 1], z=X_tsne[:, 2],
    color=['Cat' if label == 0 else 'Dog' for label in knn._y],
    labels={'x': 'TSNE1', 'y': 'TSNE2', 'z': 'TSNE3'},
    title='t-SNE 3D Visualization (Plotly)',
    opacity=0.7
)

fig.update_layout(scene=dict(
    xaxis_title='TSNE1',
    yaxis_title='TSNE2',
    zaxis_title='TSNE3'
))
fig.show()

from sklearn.manifold import TSNE

# Reduce to 3D
tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(feature_norm)

# Plot with Plotly
fig = px.scatter_3d(
    x=X_tsne[:, 0], y=X_tsne[:, 1], z=X_tsne[:, 2],
    color=['Cat' if label == 0 else 'Dog' for label in knn._y],
    labels={'x': 'TSNE1', 'y': 'TSNE2', 'z': 'TSNE3'},
    title='t-SNE 3D Visualization (Plotly)',
    opacity=0.7
)

fig.update_layout(scene=dict(
    xaxis_title='TSNE1',
    yaxis_title='TSNE2',
    zaxis_title='TSNE3'
))
fig.show()

