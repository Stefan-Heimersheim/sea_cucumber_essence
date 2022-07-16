# sea_cucumber_essence
Experiments with CNN features and circuits
## Topic
This image activates node `4` in the `block5_conv4` layer of `VGG19`. Why does it look like `sea_cucumber`?

![node4](https://github.com/Stefan-Heimersheim/sea_cucumber_essence/blob/main/node4.png?raw=true)
## Backstory
Using my [feature extraction](https://github.com/Stefan-Heimersheim/tensorflow-feature-extraction-tutorial/) script I analyzed 
node `4` in the `block5_conv4` layer of `VGG19`:
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image 
```

```python
base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
target_layer="block5_conv4"
target_index=4
steps=100
step_size=0.1
# Take the network and cut it off at the layer we want to analyze,
# i.e. we only need the part from the input to the target_layer.
target = [base_model.get_layer(target_layer).output]
part_model = tf.keras.Model(inputs=base_model.input, outputs=target)

```

```python
# The next part is the function to maximize the target layer/node by
# adjusting the input, equivalent to the usual gradient descent but
# gradient ascent. Run an optimization loop:
activation = None
@tf.function(
    # Decorator to increase the speed of the gradient_ascent function
    input_signature=(
      tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
      tf.TensorSpec(shape=[], dtype=tf.int32),
      tf.TensorSpec(shape=[], dtype=tf.float32),)
)
def gradient_ascent(img, steps, step_size):
    loss = tf.constant(0.0)
    for n in tf.range(steps):
        # As in normal NN training, you want to record the computation
        # of the forward-pass (the part_model call below) to compute the
        # gradient afterwards. This is what tf.GradientTape does.
        with tf.GradientTape() as tape:
            tape.watch(img)
            # Forward-pass (compute the activation given our image)
            activation = part_model(tf.expand_dims(img, axis=0))
            print(activation)
            print(np.shape(activation))
            # The activation will be of shape (1,N,N,L) where N is related to
            # the resolution of the input image (assuming our target layer is
            # a convolutional filter), and L is the size of the layer. E.g. for a
            # 256x256 image in "block4_conv1" of VGG19, this will be
            # (1,32,32,512) -- we select one of the 512 nodes (index) and
            # average over the rest (you can average selectively to affect
            # only part of the image but there's not really a point):
            loss = tf.math.reduce_mean(activation[:,:,:,target_index])

        # Get the gradient, i.e. derivative of "loss" with respect to input
        # and normalize.
        gradients = tape.gradient(loss, img)
        gradients /= tf.math.reduce_std(gradients)
    
        # In the final step move the image in the direction of the gradient to
# increate the "loss" (our targeted activation). Note that the sign here
# is opposite to the typical gradient descent (our "loss" is the target 
# activation which we maximize, not something we minimize).
        img = img + gradients*step_size
        img = tf.clip_by_value(img, -1, 1)
    return loss, img
```

```python
# Preprocessing of the image (converts from [0..255] to [-1..1]
starting_img = np.random.randint(low=0,high=255,size=(224,224,3), dtype=np.uint8)
img = tf.keras.applications.vgg19.preprocess_input(starting_img)
img = tf.convert_to_tensor(img)
# Run the gradient ascent loop
loss, img = gradient_ascent(img, tf.constant(steps), tf.constant(step_size))
# Convert back to [0..255] and return the new image
img = tf.cast(255*(img + 1.0)/2.0, tf.uint8)
plt.imshow(np.array(img))
im = Image.fromarray(np.array(img))
im.save("node4.png")
```

## The confusing part
