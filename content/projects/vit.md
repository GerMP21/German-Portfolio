---
title: "Visual Transformer from Scratch for Pneumonia Detection"
type: page
---

## Abstract
This code implements a Vision Transformer (ViT) model for image classification, comprising custom layers such as PreNorm, Multi-Layer Perceptron (MLP), Attention, and Transformer, along with data preprocessing functions. The ViT model architecture consists of patch embedding, positional embedding, multi-head self-attention mechanism, and a multi-layer perceptron for feature extraction and classification. The code also includes functions for loading and preprocessing image data from a given dataset. Training is performed using Sparse Categorical Crossentropy loss and Adam optimizer, with validation accuracy monitored during training. The trained model achieves a validation accuracy of 81.25% after the first epoch and is evaluated on the test dataset, achieving an accuracy of 73.72%.

## Imports
``` python
%pip install einops
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
from  tensorflow.keras import layers
from tensorflow import einsum
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange
import cv2
```

## Helper Function
This function will be used to make sure that input dimensions are represented as tuples (height, width) when needed.

``` python
def pair(t):
    return t if isinstance(t, tuple) else (t, t)
```

## PreNorm Layer
Custom layer representing the Pre-Normalization used within the transformer model. It takes the following parameter:

* **fn**: The function to be applied to the normalized input. In the transformer, this function can be either the attention mechanism or the MLP.

The call method is where the actual pre-normalization takes place. It takes the following parameters:

* **x**: The input tensor passed through the layer normalization.
* **training**: Used to enable/disable dropout layers based on the training mode.

``` python
class PreNorm(Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()

        self.norm = layers.LayerNormalization()
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(self.norm(x), training=training)

```

## Multi-Layer Perceptron Layer
Custom layer representing the Multi-Layer Perceptron used within the transformer model. It takes the following parameters:

* **dim**: The output dimension of the MLP layer, which is also the input and output dimension of each transformer block in the ViT model.
* **hidden_dim**: The dimension of the hidden layer in the MLP. It determines the intermediate dimension between the input and output of the two dense layers.
* **dropout**: The dropout rate applied to the output of both dense layers in the MLP. By default, it is set to 0.0 (no dropout).

The call method takes the following parameters:

* **x**: The input tensor that is processed through the MLP.
* **training**: Used to enable/disable dropout layers based on the training mode.

The GELU activation function has two implementations: the approximate version and the exact version. The approximate flag can be set to True to use the approximate GELU. By default, the exact GELU is used.

``` python
class MLP(Layer):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(MLP, self).__init__()

        def GELU():
            def gelu(x, approximate=False):
                if approximate:
                    coeff = tf.cast(0.044715, x.dtype)
                    return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
                else:
                    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

            return layers.Activation(gelu)

        self.net = Sequential([
            layers.Dense(units=hidden_dim),
            GELU(),
            layers.Dropout(rate=dropout),
            layers.Dense(units=dim),
            layers.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)

```

## Attention Layer
Custom layer representing the attention mechanism used in the transformer model. It takes the following parameters:

* **dim**: The input and output dimension of the attention layer.
* **heads**: The number of attention heads.
* **dim_head**: The dimension of each attention head.
* **dropout**: The dropout rate applied to the attention weights.

The call method is where the actual attention calculation takes place. It takes the following parameters:

* **x**: The input tensor passed through the layer normalization.
* **training**: Used to enable/disable dropout layers based on the training mode.

The Attention class uses two sub-layers:

* **self.attend**: This is the softmax activation function, which calculates the attention weights using the dot products between queries and keys. The softmax ensures that the attention weights are normalized and sum up to 1.
* **self.to_qkv**: This is a linear transformation layer without biases, projecting the input tensor x to the queries, keys, and values. The output dimension of this layer is inner_dim * 3, where inner_dim is the dimension of queries, keys, and values for multi-head attention.
* **self.to_out**: This is a list of layers used to project the attention output back to the original input dimension dim, followed by dropout. If project_out is False (which happens when there is only one attention head and its dimension is the same as dim), this layer is an empty list, indicating that no additional projection is needed.

``` python
class Attention(Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = layers.Softmax()
        self.to_qkv = layers.Dense(units=inner_dim * 3, use_bias=False)

        if project_out:
            self.to_out = [
                layers.Dense(units=dim),
                layers.Dropout(rate=dropout)
            ]
        else:
            self.to_out = []

        self.to_out = Sequential(self.to_out)

    def call(self, x, training=True):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        # x = tf.matmul(attn, v)
        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x, training=training)

        return x
```

## Transformer Layer
Custom layer representing the core building block of the transformer model. It takes the following parameters:

* **dim**: The output dimension of the transformer block.
* **depth**: The number of transformer blocks to stack.
* **heads**: The number of attention heads in the multi-head attention mechanism.
* **dim_head**: The dimension of each attention head. The total dimension of queries, keys, and values will be dim_head * heads.
* **mlp_dim**: The dimension of the hidden layer in the MLP used within the transformer block.
* **dropout**: The dropout rate applied to the output of both attention and MLP layers in the transformer block.

The call method is where the input tensor x is processed through the transformer blocks. It takes the following parameters:

* **x**: The input tensor passed through the layer normalization.
* **training**: Used to enable/disable dropout layers based on the training mode.

``` python
class Transformer(Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(Transformer, self).__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([
                PreNorm(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(MLP(dim, mlp_dim, dropout=dropout))
            ])

    def call(self, x, training=True):
        for attn, mlp in self.layers:
            x = attn(x, training=training) + x
            x = mlp(x, training=training) + x

        return x
```

## Visual Transformer Model
Custom model representing the Vision Transformer model. It takes the following parameters:

* **image_size**: The size of the input image. If you have rectangular images, the image_size should be the maximum of the width and height to maintain aspect ratio.
* **patch_size**: The size of each patch in the image. The image_size must be divisible by patch_size.
* **num_classes**: The number of classes to classify. It represents the output dimension of the final classification layer.
* **dim**: The output dimension of the transformer block. This is usually the hidden dimension of the transformer.
* **depth**: The number of transformer blocks to stack.
* **heads**: The number of attention heads in the multi-head attention mechanism.
* **mlp_dim**: The dimension of the hidden layer in the MLP used within the transformer block.
* **pool**: The pooling type for obtaining the final classification. It can be either 'cls' (using the class token) or 'mean' (using mean pooling).
* **dim_head**: The dimension of each attention head. The total dimension of queries, keys, and values will be dim_head * heads.
* **dropout**: The dropout rate applied to the output of both attention and MLP layers in the transformer block. By default, it is set to 0.0 (no dropout).
* **emb_dropout**: The embedding dropout rate. It is applied to the output of the patch embeddings and the positional embeddings.

The call method is the forward pass of the model. It takes the following parameters:

* **img**: This is the input image tensor that will be passed through the ViT model.
* **training**: This is a boolean argument that controls whether the model is in training mode or not. It is used to enable or disable certain operations, such as dropout layers, based on the training status. By default, it is set to True, indicating that the model is in training mode.

The shape of img should be (batch_size, image_height, image_width, num_channels), where:

* **batch_size**: The number of input images in a batch.
* **image_height**: The height of the input image.
* **image_width**: The width of the input image.
* **num_channels**: The number of channels in the input image (e.g., 3 for RGB images).

``` python
class ViT(Model):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, dropout=0.0, emb_dropout=0.0):
        super(ViT, self).__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embedding = Sequential([
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            layers.Dense(units=dim)
        ], name='patch_embedding')

        self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, num_patches + 1, dim]))
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, dim]))
        self.dropout = layers.Dropout(rate=emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.mlp_head = Sequential([
            layers.LayerNormalization(),
            layers.Dense(units=num_classes)
        ], name='mlp_head')

    def call(self, img, training=True, **kwargs):
        x = self.patch_embedding(img)
        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x, training=training)

        x = self.transformer(x, training=training)

        if self.pool == 'mean':
            x = tf.reduce_mean(x, axis=1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)

        return x
```

## Load Dataset

``` python
def read_and_preprocess_images(image_folder, target_size):
    images = []
    labels = []

    for label, class_name in enumerate(os.listdir(image_folder)):
        class_folder = os.path.join(image_folder, class_name)
        for image_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, target_size)  # Resize all images to a target size
            image = image.astype(np.float32) / 255.0  # Normalize the image pixel values
            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)
```

``` python
# Define the image size for ViT
image_size = (128, 128)  # You can choose an appropriate size based on the images

# Define the paths to the train and test folders
train_folder = "/kaggle/input/chest-xray-pneumonia/chest_xray/train"
val_folder = "/kaggle/input/chest-xray-pneumonia/chest_xray/val"
test_folder = "/kaggle/input/chest-xray-pneumonia/chest_xray/test"

# Read and preprocess images from the train and test folders
train_images, train_labels = read_and_preprocess_images(train_folder, image_size)
val_images, val_labels = read_and_preprocess_images(val_folder, image_size)
test_images, test_labels = read_and_preprocess_images(test_folder, image_size)

# Convert labels to one-hot encoded vectors
num_classes = 2
train_labels = tf.one_hot(train_labels, depth=num_classes)
val_labels = tf.one_hot(val_labels, depth=num_classes)
test_labels = tf.one_hot(test_labels, depth=num_classes)

# Create a dataset from the training images and labels
batch_size = 32  # You can choose an appropriate batch size based on your memory capacity
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=train_images.shape[0]).batch(batch_size)
```

## Training

``` python
def train_model(model, train_dataset, val_images, val_labels, epochs, batch_size):
    # Create a SparseCategoricalCrossentropy loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Create an Adam optimizer with a learning rate of 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Define a train_step function, which will be called in each training iteration
    def train_step(images, labels):
        # Create a GradientTape to compute gradients for trainable variables
        with tf.GradientTape() as tape:
            # Make predictions using the model with training=True to enable dropout, etc.
            predictions = model(images, training=True)
            # Compute the loss between the predicted values and the actual labels
            loss = loss_fn(labels, predictions)

        # Calculate gradients of the loss with respect to trainable variables
        gradients = tape.gradient(loss, model.trainable_variables)
        # Apply the gradients to update the model's trainable variables
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Return the loss for this training step
        return loss

    # Training loop: run for the specified number of epochs
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # Print the number of epochs
        print(f'Epoch {epoch + 1}/{epochs}')

        # Iterate over the training dataset in batches
        for batch_images, batch_labels in train_dataset:
            # Perform a training step for the current batch and get the loss
            # Convert one-hot encoded labels to integer labels
            labels = tf.argmax(batch_labels, axis=-1)
            loss = train_step(batch_images, labels)
            # Accumulate the loss for this epoch
            total_loss += loss
            num_batches += 1

        # Calculate the average loss for this epoch
        average_loss = total_loss / num_batches
        
        val_predictions = vit(val_images, training=False)
        val_accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(val_labels, val_predictions))
        
        # Print the average loss of the model
        print(f'Train Loss: {average_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy.numpy():.4f}')
        
        if val_accuracy.numpy() > 0.80:
            break

    # Return the trained model
    return model
```

``` python
# Define the ViT model
vit = ViT(
    image_size=image_size,
    patch_size=16,  # You can choose an appropriate patch size based on the image size and complexity
    num_classes=num_classes,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
)

# Train the model on the new dataset
epochs = 10
vit = train_model(vit, train_dataset, val_images, val_labels, epochs, batch_size)
```
> Epoch 1/10
>
> Train Loss: 0.5387
>
> Val Accuracy: 0.8125

## Test Model

``` python
# Evaluate the model on the test dataset
test_predictions = vit(test_images, training=False)
test_accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(test_labels, test_predictions))
print(f'Test Accuracy: {test_accuracy.numpy():.4f}')
```
> Test Accuracy: 0.7372