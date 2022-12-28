# Neural style transfer experiments at sustainable human scale

**Important note:** this page, *and this repository more generally*, are being constantly updated as experiments end and are ready to be posted, and so will change considerably as new material is added.

## Introduction

**Abstract.** In what follows I will report, in long form and with various Python scripts and Jupyter Notebooks, on experiements with transfer learning and more precisely neural style transfer. I will restrict to the scenario of transforming a photograph to *look like* a famous painting (Van Gogh's *The Starry Night* unless otherwise noted). The interest is at once 
- practical: running times of various algorithms, optimization schemes, etc. on *normal/old hardware* 
- and abstract: how pleasant and usable the results are. 

I will use these two aspects as soft metrics throughout.

**Idea, use case scenario, and some questions.** Suppose an artist, *owning normal or even past-its-prime hardware* (perhaps a 7 or 10 year-old Macbook Pro/Air, perhaps a 3-4 year-old PC), would like to add neural style transfer to their techniques. Is this possible given the contraints? Is this feasible? Can subjectively interesting results be achieved given enough time? Is the investment in learning neural networks and using a machine learning platform worth it despite the possible impracticality of the actual algorithms and uselessness of the results generated? That is, can transfer learning be achived at the human level and the techniques be useful somewhere else down the pipeline? (The last question obviously transcends whether the user is an artist.) Is is sustainable? (Is it environmentally sustainable? Is electricity pricing making this prohibitive to do at home? Is it resource intensive to the point one has to dedicate a computer solely to the task?) 

**Objectives.** Here are two important objectives guiding the experiments:

- achieve interesting (for some *reasonable definition of interesting*) results on a human-scale budget, on middle-to-low-level hardware (could be years old), perhaps without using GPU training or even CPU parallelization;
- achieve human-scale *displayable and printable* images; note that there is a large difference between the two image scales here, and my experiments will for the most part use images of **size 533x400** width times height, which is the *display scale*. The *printable scale* is two orders of magnitude (100) times bigger (10x in each direction): for a 30x40 cm quality print, one is looking at images of 20+ megapixels, or gigantic resolutions on the order of **6000x4500** pixels. The printable scale is perhaps beyond any reasonable machine learning algorithm for the moment (late 2022), at least without any serious upsampling or other heavy pre/post-processs image manipulation tricks, and this scale is certainly beyond what any *human-scale computer* (i.e. one as described in the previor paragraph) can currently do in a matter of hours or a few days.

## Requirements

On the software side, the following standard packages are used alongside Python 3:

- ```tensorflow```
- ```numpy```
- ```PIL``` (Python imaging library)
- ```matplotlib```
- ```imagemagick``` (to make gifs or do other image conversions)

On the scientific side, I assume enough knowledge of machine learning and linear algebra to start experimentation. A quick basic tutorial is [this great website by Harrish Narayanan](https://github.com/titu1994/Neural-Style-Transfer). A very brief introduction preaching to the choir (i.e. for people already familiar with machine learning) is given below.

On the artistic side, I assume nothing, and what perhaps looks interesting to me looks completely bizarre to a million other randomly sampled people.

## Brief introduction to (neural style) transfer learning (under construction)

To quote François Chollet from the [Keras webpage](https://keras.io/guides/transfer_learning/) explaining the subject,

> transfer learning consists of taking features learned on one problem, and leveraging them on a new, similar problem. For instance, features from a model that has learned to identify racoons may be useful to kick-start a model meant to identify tanukis.

For us models mean (convolutional) neural networks, usually trained on the ImageNet database. Two such examples are the VGG16 and VGG19 networks of Simonyan and Zisserman referenced below. They can be found in the [Keras applications API](https://keras.io/api/applications/vgg/).

Neural style transfer (NST) is a form of transfer learning originally introduced in the paper of Gatys et al referenced below. For a detailed textbook description, see Chollet's *Deep learning with Python*, second edition, from Manning Press (referenced below). See also the article *Convolutional neural networks for artistic style transfer* by Harrish Narayanan (from his website, referenced below).

In its simplest form, NST takes a content image $C$, and via a sequence of manipulations, it transforms it into a generated image $G$ by trying to *add the style* of another image, the style image $S$. This being machine learning with neural networks, $C,S,G$ are to be thought of as vectors (or tensors if you wish). The problem is then to find, after a number of iterations of the learning algorithm, the image $G$ which maximizes the following loss function:

$$ L(G) = \alpha \cdot L_c(C, G) + \beta \cdot L_s(G, S) + \gamma \cdot L_{TV} (G) $$

with $L_c(C, G)$ the *content loss function*, $L_s(G, S)$ the *style loss function*, and $L_{TV} (G)$ the *total variation (continuity) loss function*. Here $\alpha, \beta, \gamma$ are positive parameters to be determined; the important ratio is $\alpha / \beta$ which measures how much we emphasize content over style (how much we want the image to look like the original $C$ versus the style $S$). 

The total variation loss function is the easiest to explain: it looks at pixels which are nearby in $G$, measuring how close their values are. It insures a sense of continuity in the resulting image, and is sometimes omitted by setting $\gamma = 0$. The content loss function measures how close $C$ is to $G$, in some internal layer of a neural network:

$$ L_c(C, G) = ||a(C) - a(G)||_2^2 $$

where $a$ is the value of the activation parameter tensor in that specific layer (when $C,G$ are fed as inputs to the neural network). Finally the main contribution of Gatys et al is the style function, which measures how close $C$ and $S$ are *across the board* at various length scales, or rather at various intermmediate and upper layers of the neural network. It does so via so-called Gram matrices, and is defined as

$$ L_s(G, S) = \sum_{\ell} \delta_\ell ||Gr_\ell(G) - Gr_\ell(S)||_F^2 $$

where the sum is over a bunch of layers $\ell$ in the neural network, $|| \cdot ||_F$ is [matrix Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm), and the Gram matrix $Gr(X)$ at level $\ell$ in the neural network is a square matrix of size $n_c \times n_c$ (here $n_c$ is the number of channels at that level) defined as 

$$Gr(X)_{kk'} = \frac{1}{(2 n_c n_h n_w)^2} \sum_{i=1}^{n_h} \sum_{j=1}^{n_w} a_{ijk}(X) a_{ijk'}(X)$$

with $a_{ijk}(X)$ the value of the activation tensor at level $\ell$ on input $X$ (either $C$ or $G$), and it usually has shape (assuming a channel last approach) $n_h \times n_w \times n_c$ where $n_h$ is the height of the image at that stage (likewise $n_w$ stands for width, $n_c$ for the number of channels). Finally $\delta_\ell$'s are weights to be chosed for the appropriate layers. Each measures how important the respective layer is.

## Models and experiments

### Experiment 1: by the book

### Experiment 2: theme and variations (under construction)

## References

- Chollet, *Deep learning with Python*, second edition, [publisher's website](https://www.manning.com/books/deep-learning-with-python-second-edition) and [GitHub repositiory](https://github.com/fchollet/deep-learning-with-python-notebooks)
- Chollet, *Neural style transfer*, [Keras.io GitHub examples repository](https://github.com/keras-team/keras-io/blob/master/examples/generative/neural_style_transfer.py)
- Chollet, *Transfer learning and fine-tuning*, [Keras website tutorial](https://keras.io/guides/transfer_learning/)
- Gatys, Ecker, Bethge, *Image style transfer using convolutional neural networks*, [link to PDF](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
- Log0, *Neural style painting*, [GitHub repository](https://github.com/log0/neural-style-painting)
- Narayanan, *Convolutional neural networks for artistic style transfer*, [website](https://harishnarayanan.org/writing/artistic-style-transfer/) and [GitHub repository](https://github.com/titu1994/Neural-Style-Transfer)
- Simonyan, Zisserman, *Very deep convolutional networks for large-scale image recognition*, [link to arXiv](https://arxiv.org/pdf/1409.1556.pdf)
