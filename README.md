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
Judging my the [OpenAI Microscope](https://microscope.openai.com/models/vgg19_caffe/conv5_4_conv5_4_0/4) it looks like the node mostly gets activated by furry animals -- _in the training set_. Of course our image in artificial and this far outside the usual distribution, and we can expect such different behaviour. But why do we get the `sea_cucumber` prediction, rather than predictions of `dog`, `bison` or `lion`?

Feeding this image into the network, it seems insanely sure that the right label is `sea_cucumber`. Also other imagenet-trained networks such as Inception or VGG16 give the same result. Note: This was not indended and not optimized for.

```python
model_vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=True)
x = tf.keras.applications.vgg19.preprocess_input(np.expand_dims(img, axis=0))
predictions = model_vgg19.predict(x)
print('Predicted:', tf.keras.applications.vgg19.decode_predictions(predictions, top=3)[0])
```
```
Predicted: [('n02321529', 'sea_cucumber', 1.0), ('n01924916', 'flatworm', 1.2730256e-33), ('n01981276', 'king_crab', 2.537045e-37)]
```

```python
model_vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
x = tf.keras.applications.vgg16.preprocess_input(np.expand_dims(img, axis=0))
predictions = model_vgg16.predict(x)
print('Predicted:', tf.keras.applications.vgg16.decode_predictions(predictions, top=3)[0])
```
```
Predicted: [('n02321529', 'sea_cucumber', 1.0), ('n01950731', 'sea_slug', 4.6657154e-15), ('n01924916', 'flatworm', 1.810621e-15)]
```

```python
model_resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=True)
x = tf.keras.applications.resnet.preprocess_input(np.expand_dims(img, axis=0))
predictions = model_resnet.predict(x)
print('Predicted:', tf.keras.applications.resnet.decode_predictions(predictions, top=3)[0])
```
```
Predicted: [('n02321529', 'sea_cucumber', 0.9790509), ('n12144580', 'corn', 0.00899157), ('n13133613', 'ear', 0.005869923)]
```

Even this online service ([snaplogic using Inception](https://www.snaplogic.com/machine-learning-showcase/image-recognition-inception-v3)) mistakes a picture of my phone screen showing the image:
![recognize](https://github.com/Stefan-Heimersheim/sea_cucumber_essence/blob/main/recognize.png?raw=true)

## Investigation
Let's look at the activations, after feeding the image into the VGG19 network I have been using:
```python
target = [model_vgg19.get_layer("block5_conv4").output]
model_vgg19_cutoff = tf.keras.Model(inputs=model_vgg19.input, outputs=target)
x = tf.keras.applications.vgg19.preprocess_input(np.expand_dims(img, axis=0))
activations = model_vgg19_cutoff.predict(x)
plt.plot(np.mean(np.mean(np.mean(activations, axis=0), axis=0), axis=0))
```
![activations](https://github.com/Stefan-Heimersheim/sea_cucumber_essence/blob/main/activations.png?raw=true)
So the question we're asking, is this the typical pattern for a dog or bison? Or maybe closer to the `sea_cucumber` pattern, in this 512-dimensional space?

Let's have a look at the `groenendael` (1st image in Microscope) and `ox` (3rd image in Microscope) classes. I downloaded the imagenet data and used [this list](https://image-net.org/challenges/LSVRC/2017/browse-synsets.php) to find the right files.

```python
def plot_activations(img_path, ax=None):
	if ax==None:
		fig, ax = plt.subplots()
	x = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(x)
	x = np.expand_dims(x, axis=0)
	x = tf.keras.applications.vgg19.preprocess_input(x)
	activations = model_vgg19_cutoff.predict(x)
	av_activations = np.mean(np.mean(np.mean(activations, axis=0), axis=0), axis=0)
	ax.plot(av_activations)
	ax.scatter(4, av_activations[4])
	ax.set_xlabel("block5_conv4 index")
	ax.set_ylabel("Activation value")
	return ax
```
```python
fig, axs = plt.subplots(nrows=3)
fig.suptitle(" n02105056: groenendael")
plot_activations("/data/nfs/ILSVRC2012_img_train/n02105056/n02105056_10005.JPEG", ax=axs[0])
plot_activations("/data/nfs/ILSVRC2012_img_train/n02105056/n02105056_10013.JPEG", ax=axs[1])
plot_activations("/data/nfs/ILSVRC2012_img_train/n02105056/n02105056_10020.JPEG", ax=axs[2])
plt.savefig("groenendael.png", dpi=600)
plt.show()
```
![groenendael](https://github.com/Stefan-Heimersheim/sea_cucumber_essence/blob/main/groenendael.png?raw=true)
Hmm I don't really see a pattern by eye here, nor a similarity to above / excitation in index 4. Let's plot something like absolute distance in this 512-dim vector space, compared to the average distance.
```python
node4max_activations = model_vgg19_cutoff.predict(tf.keras.applications.vgg19.preprocess_input(np.expand_dims(img, axis=0)))

import glob
groenendael_images = glob.glob("/data/nfs/ILSVRC2012_img_train/n02105056/*.JPEG")
def get_activations(img_path):
	x = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(x)
	x = np.expand_dims(x, axis=0)
	x = tf.keras.applications.vgg19.preprocess_input(x)
	activations = model_vgg19_cutoff.predict(x)
	return activations

groenendael_activations = [get_activations(i) for i in groenendael_images]
# First 200 of 1st category, for comparison
tench_images = glob.glob("/data/nfs/ILSVRC2012_img_train/n02105056/*.JPEG")[:200]
tench_activations = [get_activations(i) for i in tench_images] #for comparison

def distance(a,b=node4max_activations):
	return np.sqrt(np.sum((a-b)**2))

groenendael_distances = [distance(a) for a in groenendael_activations]
tench_distances = [distance(a) for a in tench_activations]

plt.hist(tench_distances, color="grey", density=True, bins=100)
plt.hist(groenendael_distances, color="red", density=True, bins=100, alpha=0.5)
plt.xlabel("Distance (L2 norm in 512 dimensions)")
plt.savefig("distances.png", dpi=600)
plt.show()
```
![distance_histogram](https://github.com/Stefan-Heimersheim/sea_cucumber_essence/blob/main/distances.png?raw=true)
Hmm, doesn't look like we can see anything. Let's compare to distance within `groenendael` class for comparison:

```python

groenendael_self_distances = [distance(a, b=groenendael_activations[0]) for a in groenendael_activations[1:]]

plt.hist(tench_distances, color="grey", density=True, bins=100)
plt.hist(groenendael_self_distances, color="blue", density=True, bins=100, alpha=0.5)
plt.xlabel("Distance (L2 norm in 512 dimensions)")
plt.savefig("distances_self.png", dpi=600)
plt.show()
```


_Note to myself: Look at patterns for similar things (various dogs / animals?) and see if they look similar? What about some clustering like [t-SNE](https://distill.pub/2016/misread-tsne/) to help us?_
