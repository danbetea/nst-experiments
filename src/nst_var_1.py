from PIL import Image
import numpy as np
import tensorflow as tf




##################################################################################
#
# global variables
#

# image dimensions throughout; ratio 4:3
img_width = 533
img_height = 400

# load VGG19 (16 would also work) model, no top 4 layers, set it to non-trainable
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_height, img_width, 3),
                                  weights='imagenet')

# we are not training the model
vgg.trainable = False

# layers and weights to use for the style loss function
# these are the weights assigned to each layer for the style cost
# high weights assigned to late layers means softer (higher level)
# features of the style image are favored
#
# NOTE: lots of optimization possible here
# 
# # one way to instantiate is for example: 
# style_layers = [
#     ('block1_conv1', 0.5),
#     ('block2_conv1', 1.0),
#     ('block3_conv1', 1.5),
#     ('block4_conv1', 2.0),
#     ('block5_conv1', 2.5)]

# we will nevertheless use a 'hat-shaped' weight vector 
style_layers = [
    ('block1_conv1', 0.5),
    ('block2_conv1', 1.0),
    ('block3_conv1', 2.0),
    ('block4_conv1', 1.0),
    ('block5_conv1', 0.5)]

# layer to use for the content loss function
# for vgg19, one can use also block5_conv4
content_layer = [('block5_conv3', 1)]   

# loading images, putting them into tensorflow constants
content_image = np.array(Image.open("../images/san_francisco_small.png").resize((img_width, img_height)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
style_image =  np.array(Image.open("../images/van_gogh_starry_night_small.png").resize((img_width, img_height)))
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

# assign weights for content and style in computing total loss
# note there is no total variation loss here
alpha = 5
beta = 100

# number of training steps
epochs = 2501

# optimizer used
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)




##################################################################################
#
# utility functions
#

def tensor_to_image(tensor):
    """
    converts tensor to image with PIL
    """
    # rescale back to max 255, convert to int array
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)

    # if in dimension 4, first element better be 1 (i.e. only one example in the batch)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def gram_matrix(A):
    """
    computes GA = A * transpose(A) with A a matrix of shape (n_C, n_H * n_W)
    """  
    GA = tf.linalg.matmul(A, tf.linalg.matrix_transpose(A))
    return GA




##################################################################################
#
# compute content and style losses
#

# first content loss

def compute_content_loss(content_output, generated_output):
    """
    computes the content cost
    
    input:
    content_output = encoding of the content image using all layers selected (style + content) in the nn
    generated_output = same as above for generated image

    output: 
    L_content = scalar that you compute using equation 1 above.
    """

    # retrieve the tensors, last element corresponds to the content layer
    a_C = content_output[-1] # of shape (1, n_H, n_W, n_C)
    a_G = generated_output[-1] # of shape (1, n_H, n_W, n_C)
        
    # retrieve the dimensions of each
    _, n_H, n_W, n_C = a_G.shape 
    
    # reshape a_C and a_G so they're (_, n_H * N_W, n_C) 
    a_C_unrolled = tf.reshape(a_C, shape=[_, -1, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[_, -1, n_C])
    
    # compute the L^2 content cost
    # NOTE: can divide by other things here
    L_content = tf.math.square(tf.norm(a_C_unrolled - a_G_unrolled))/(4*n_H*n_W*n_C)
        
    return L_content

# next style loss in a few steps

def compute_layer_style_loss(a_S, a_G):
    """
    input:
    a_S = tensor of dimension (1, n_H, n_W, n_C), encoding of style image S using hidden layer activations 
    a_G = tensor of dimension (1, n_H, n_W, n_C), same as above, for generated image
    
    output: 
    L_style_layer = style cost for a specific layer
    """
    
    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # reshape so that final shape is (n_C, n_H * n_W)
    a_S = tf.transpose(a_S, perm=[0, 3, 1, 2]) # shape is now (_, n_C, n_H, n_W)
    a_G = tf.transpose(a_G, perm=[0, 3, 1, 2]) # shape is now (_, n_C, n_H, n_W)
    a_S = tf.reshape(a_S, shape=[n_C, n_H * n_W])
    a_G = tf.reshape(a_G, shape=[n_C, n_H * n_W])

    # the actual loss
    G_S = gram_matrix(a_S)
    G_G = gram_matrix(a_G)
    L_style_layer = tf.math.divide( \
                                   tf.math.reduce_sum(tf.math.square(tf.math.subtract(G_S, G_G))), \
                                   (4 * (n_C**2.) * (n_H**2.) * (n_W**2.)) \
                                  )    
    
    return L_style_layer

# computes overall style loss

def compute_style_loss(style_image_output, generated_image_output, style_layers=style_layers):
    """
    computes the overall style cost from several chosen layers
    
    input:
    style_image_output = our tensorflow model
    generated_image_output =
    style_layers = list containing: (layer name used, a coefficient for each of them)
    
    output: 
    L_style = the style loss
    """
    
    L_style = 0

    # set a_S = hidden layer activation for style image from the layers selected, concatenated
    # the last element contains the content layer image, which we don't want
    # same for a_G, but for generated image
    a_S = style_image_output[:-1]
    a_G = generated_image_output[:-1]

    for i, weight in zip(range(len(a_S)), style_layers):  
        # style loss for the current layer
        L_style_layer = compute_layer_style_loss(a_S[i], a_G[i])
        # added with the correct weight
        L_style += weight[1] * L_style_layer

    return L_style

# computes total loss

# compiled, so it runs fast
@tf.function()
def total_loss(L_content, L_style, alpha = 10, beta = 40):
    """
    computes the total cost function L = alpha * L_content + beta * L_style
    """
    L = alpha * L_content + beta * L_style
    return L




##################################################################################
#
# initialize generated image
#

# instantiate the generated_image as the content_image
generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
# add some noise to the generated_image to make convergence faster
# NOTE: lots of optimization can go into the noise values here
noise = tf.random.uniform(tf.shape(generated_image), -0.2, 0.2)
generated_image = tf.add(generated_image, noise)
# clip the final result and convert into variable (again!)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
generated_image = tf.Variable(generated_image)




##################################################################################
#
# style and content encoders
#

def get_layer_outputs(vgg, layer_names):
    """ 
    returns a list of model's (vgg16 or 19) intermediate layers output values
    """
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

vgg_model_outputs = get_layer_outputs(vgg, style_layers + content_layer)

# set a_C and a_S, used for content and style  
# using content and style layers, from content and style images
preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)
preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)




##################################################################################
#
# training step function
#

# compiled, so it runs fast
@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:

        # get the encoding of the generated image
        a_G = vgg_model_outputs(generated_image)
        
        # compute style, content, and total cost
        L_style = compute_style_loss(a_S, a_G)
        L_content = compute_content_loss(a_C, a_G)
        L = total_loss(L_content, L_style, alpha = alpha, beta = beta)
    
    # do gradient descent
    #
    # here we do 
    #
    #          g = g - learning_rate * grad      with g = generated image

    # get the gradients    
    grad = tape.gradient(L, generated_image)
    # update the generated image by applying gradient descent
    optimizer.apply_gradients([(grad, generated_image)])
    # renormalize
    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))
    # useful to return the loss
    return L




##################################################################################
#
# the actual training
#

for i in range(epochs):
    L = train_step(generated_image)
    # comment out if you don't want to look at each individual step
    print(f"Iteration {i:04}: loss={L:.4f}")
    if i % 100 == 0:
        # save the generated image at regular intervals
        image = tensor_to_image(generated_image)
        image.save(f"../images/output/square_image_at_{i:04}.jpg")