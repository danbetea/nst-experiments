# Neural style transfer experiments at sustainable human scale

**Important note:** this page, *and this repository more generally*, are being constantly updated as experiments end and are ready to be posted, and so will change considerably as new material is added.

## Few line description

In this repository and on this page I will address the following questions: **how** can one transform an image (e.g. a random photograph) like the one on the left below[^1] to look like it was painted in the style of the middle image (Van Gogh's "The Starry Night"), perhaps obtaining a gif[^2] of the transformation like the one in the image below on the right? **How long** does it take to produce such results on regular laptop hardware? **What kind of results** are even possible?

<table>
  <tc>
    <td> <img src="/images/san_francisco_small.png"  alt="content" width = 240px height = 180px ></td>
  </tc> 
  <tc>
    <td><img src="/images/van_gogh_starry_night_small.png" alt="style" width = 240px height = 180px></td>
  </td>
  </tc>
  <tc>
    <td><img src="/images/animation_chollet_book_exp1_r1.gif" alt="generated animation" width = 240px height = 180px></td>
  </td>
  </tc>
</table>

[^1]: For the sake of reproducibility I am using the same source image as in (Section 12.3 of the) the book by François Chollet, *Deep learning with Python*, Manning Press, second edition. It is one of the standard texts on deep learning algorithms with Python and Tensorflow/Keras, and the only one I know of discussing the techniques in this report.
[^2]: If the gif is not animated, please click on it.

## Introduction

**Brief description.** In what follows I will report, in long form and with various Python scripts and Jupyter Notebooks, on experiements with transfer learning and more precisely neural style transfer. I will restrict to the scenario of transforming a photograph to *look like* a famous painting (Van Gogh's *The Starry Night* unless otherwise noted). The interest is at once 
- practical: running times of various algorithms, optimization schemes, etc. on *normal/old hardware* 
- and abstract: how pleasant and usable the results are. 

I will use and *report* these two aspects as soft metrics throughout.

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

On the scientific side, I assume enough knowledge of machine learning and linear algebra to start experimentation. A quick basic tutorial is [this great website by Harish Narayanan](https://github.com/titu1994/Neural-Style-Transfer). A very brief introduction preaching to the choir (i.e. for people already familiar with machine learning) is given below.

On the artistic side, I assume nothing, and what perhaps looks interesting to me looks completely bizarre to a million other randomly sampled people.

## Hardware

The following hardware has been used for the experiments

- 2019 HP EliteBook 735 G6 14", 32 GB of RAM, AMD Ryzen 5 Pro 3500U CPU, Ubuntu 22.04
- 2015 Macbook Pro Retina 13", 8GB of RAM, Inter Core i5 CPU, MacOS Big Sur

and *in practice the HP laptop has been used* for running the code more often than not as it is slightly faster and less useful in day-to-day activities (in no small part due to poor battery life).

**Remark.** Things may change here and these changes will be reported as I upgrade computers, or perhaps migrate some experiments in the cloud, etc.

## Brief introduction to (neural style) transfer learning

To quote François Chollet from the [Keras webpage](https://keras.io/guides/transfer_learning/) explaining the subject,

> transfer learning consists of taking features learned on one problem, and leveraging them on a new, similar problem. For instance, features from a model that has learned to identify racoons may be useful to kick-start a model meant to identify tanukis.

For us models mean (convolutional) neural networks, usually trained on the ImageNet database. Two such examples are the VGG16 and VGG19 networks of Simonyan and Zisserman referenced below. They can be found in the [Keras applications API](https://keras.io/api/applications/vgg/).

Neural style transfer (NST) is a form of transfer learning originally introduced in the paper of Gatys et al referenced below. For a detailed textbook description, see Chollet's *Deep learning with Python*, second edition, from Manning Press (referenced below). See also the article *Convolutional neural networks for artistic style transfer* by Harish Narayanan (from his website, referenced below).

In its simplest form, NST takes a content image $C$, and via a sequence of manipulations, it transforms it into a generated image $G$ by trying to *add the style* of another image, the style image $S$. This being machine learning with neural networks, $C,S,G$ are to be thought of as vectors (or tensors if you wish). The problem, posed as an optimization (minimization) problem by Gatys et al in a simplified form (with $\gamma=0$ below, see references) is  as follows. Find, after a number of iterations of the learning algorithm, the image $G$ which minimizes the following loss function:

$$ L(G) = \alpha \cdot L_c(C, G) + \beta \cdot L_s(G, S) + \gamma \cdot L_{TV} (G) $$

with $L_c(C, G)$ the *content loss function*, $L_s(G, S)$ the *style loss function*, and $L_{TV} (G)$ the *total variation (continuity) loss function*. Here $\alpha, \beta, \gamma$ are positive parameters to be determined; the important ratio is $\alpha / \beta$ which measures how much we emphasize content over style (how much we want the image to look like the original $C$ versus the style $S$). 

The total variation loss function is the easiest to explain: it looks at pixels which are nearby in $G$, measuring how close their values are. It ensures a sense of continuity in the resulting image, and is sometimes omitted by setting $\gamma = 0$ (Gatys et al do not even mention it). The content loss function measures how close $C$ is to $G$, in some internal layer of a *pre-trained neural network* like VGG16 or VGG19:

$$ L_c(C, G) = ||a(C) - a(G)||_2^2 $$

where $a$ is the value of the activation parameter tensor in that specific layer (when $C,G$ are fed as inputs to the neural network). Finally the main contribution of Gatys et al is the style function, which measures how close $C$ and $S$ are *across the board* at various length scales, or rather at various intermmediate and upper layers of the neural network. It does so via so-called Gram matrices, and is defined as

$$ L_s(G, S) = \sum_{\ell} \delta_\ell ||Gr_\ell(G) - Gr_\ell(S)||_F^2 $$

where the sum is over a bunch of layers $\ell$ in the neural network, $|| \cdot ||_F$ is [matrix Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm), and the Gram matrix $Gr(X)$ at level $\ell$ in the neural network is a square matrix of size $n_c \times n_c$ (here $n_c$ is the number of channels at that level) defined as 

$$Gr(X)_{kk'} = \frac{1}{(2 n_c n_h n_w)^2} \sum_{i=1}^{n_h} \sum_{j=1}^{n_w} a_{ijk}(X) a_{ijk'}(X)$$

with $a_{ijk}(X)$ the value of the activation tensor at level $\ell$ on input $X$ (either $C$ or $G$), and it usually has shape (assuming a channel last approach) $n_h \times n_w \times n_c$ where $n_h$ is the height of the image at that stage (likewise $n_w$ stands for width, $n_c$ for the number of channels). Finally $\delta_\ell$'s are weights to be chosed for the appropriate layers. Each measures how important the respective layer is.

**Remark.** The pre-trained neural network here (for the sake of clarity, one of VGG16 or VGG19) is used as a distance function computing machine. That is, it is used to compute the loss function, and it is sometimes referred to as the *loss neural network*.

**Remark.** In the description above, NST takes $C$ and $S$ as inputs, and manipulates $C$ into a successive number of $G$'s until some convergence is reached. One can equally start with a random noise image $X$ instead of $C$ and do the procedure just described, or with a combination of $C+X$ for some random noise image $X$. More importantly, one can *and should* experiment with all sorts of starting points as far as esthetics is concerned if time and other resources permit. 

## Models and experiments

### Experiment 1: by the book (under construction)

The first experiment, a base line as it were, consists of François Chollet's long description of the Gatys et al paper as given in Section 12.3 of his book *Deep learning with Python* (second edition). The relevant file is the Jupyter notebook ```nst_orig_chollet_book.ipynb```. The network used in the text, and the one in the code, is the VGG19 network of Symonian and Zisserman. All parameters for the original run, including the number of iterations (4000) were left unchanged. In particular the weights for style, content, and total variation are as follows:

$$ \alpha = 2.5 \cdot 10^{-8}, \beta = 10^{-6}, \gamma = 10^{-6}. $$

Other runs were made at 1000 training step iterations, with both VGG19 and VGG16 (changing from one to the other is a matter of simply replacing 16 by 19 in the cell loading the network). The optimizer was always ```SGD``` with a massive (for my intuition) learning rate of 100 (decreasing every 100 steps). 

#### Time measurements

In terms of timing I can report the following:

- the difference between VGG16 and VGG19 is somewhat significant:

| Time per 100 training steps | Network   |
|-----------------------------|-----------|
| approx 15 min               | VGG16     |
| approx 18 min               | VGG19     |

- otherwise, absent any parallelization, things are slow, at least measured against the human perception of "computers being fast":

| Total time           | Network | Number of training steps | Run no. |
|----------------------|---------|--------------------------|---------|
| approx 12h           | VGG19   | 4000                     | 1       |
| approx 2h 30min      | VGG16   | 1000                     | 2       |
| approx 2h 42min      | VGG16   | 1000                     | 3       |
| approx 2h 43min      | VGG16   | 1000                     | 4       |

Runs 3 and 4 have had the weights $\alpha, \beta, \gamma$ changed a bit. For example, in Run 4 $\alpha = 10^{-9}$ and $\gamma = 10^{-10}$ so I de-emphasize content and neglect continuity in the image. In terms of running time it makes little difference.

Here are some remarks:

- that this algorithm is slow absent any parallelization has been reported in every source I could find on the matter (and in almost all of the cited references). Just how slow things were was unclear until now;
- parameter tuning seems out of reach with this method;
- it is difficult to decrease the learning rate and still have the gradient descent algorithm *actually decrease the loss function*, so it seems this choice of parameters $\alpha, \beta, \gamma$ in combination with the learning rate (initially 100) is rather rigid;
- *visually*, results similar to the final result seem to be achieved after fewer than 1000 iterations, perhaps as low as 500 or 600.

#### Artistic measurements

How interesting are the results? First, despite the vast difference in running times between run 1 and run 4, the results are similar. Nevertheless one can indeed see more of the style after a longer number of iterations as exemplified below: side by side are the results of VGG19 at 4000 iterations (run 1 above, on the left) and of VGG16 at 1000 iterations and with different than the original $\alpha, \gamma$ (run 4 above, on the right):

<table>
  <tc>
    <td> <img src="/images/book_vgg19_4000.png"  alt="run 1 image" width = 533px height = 400px ></td>
  </tc> 
  <tc>
    <td><img src="/images/book_vgg16_1000.png" alt="run 4 image" width = 533px height = 400px></td>
  </td>
  </tc>
</table>

#### Conclusion for experiment 1

Experiment 1, by the book, is rather slow to run, and results are interesting but not particularly pleasant to the eye. Cholet acknowledges both points in his book, and further claims one should expect no miracles with these hyperparameters. He further the approach to more of a signals processing (noising, denoising, sharpening, etc.) approach than a true deep learning approach, and certainly the results do not contradict his claims. However there are many hyperparameters or blocks of code that can be switched, and the results are promising enough to pursue experimentation.

Finally, in the course of testing various hyperparameters that failed to make the training iterations converge (minimize the loss function), I accidentally saved one of the resulting images, after just *one* iteration. The result is below. Now *this is interesting*, much more so than the above, *to my eyes* (perhaps except the red tint, easily removable in post processing).

<img src="/images/book_vgg16_accident.png" alt="run 4 image" width = 533px height = 400px>

### Experiment 2: theme and variations (under construction)

## References

- Chollet, *Deep learning with Python*, second edition, [publisher's website](https://www.manning.com/books/deep-learning-with-python-second-edition) and [GitHub repositiory](https://github.com/fchollet/deep-learning-with-python-notebooks)
- Chollet, *Neural style transfer*, [Keras.io GitHub examples repository](https://github.com/keras-team/keras-io/blob/master/examples/generative/neural_style_transfer.py)
- Chollet, *Transfer learning and fine-tuning*, [Keras website tutorial](https://keras.io/guides/transfer_learning/)
- Gatys, Ecker, Bethge, *Image style transfer using convolutional neural networks*, [link to PDF](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), [link to arXiv](https://arxiv.org/abs/1508.06576)
- Johnson, Alahi, Fei-Fei, *Perceptual losses for real-time style transfer and super-resolution*, [link to arXiv](https://arxiv.org/abs/1603.08155), [publication and supplementary material](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_43)
- Log0, *Neural style painting*, [GitHub repository](https://github.com/log0/neural-style-painting)
- Narayanan, *Convolutional neural networks for artistic style transfer*, [website](https://harishnarayanan.org/writing/artistic-style-transfer/) and [GitHub repository](https://github.com/titu1994/Neural-Style-Transfer)
- Simonyan, Zisserman, *Very deep convolutional networks for large-scale image recognition*, [link to arXiv](https://arxiv.org/pdf/1409.1556.pdf)
