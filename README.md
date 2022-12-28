# Neural style transfer experiments at human scale

## Introduction

**Abstract.** In what follows I will report, in long form and with various Python scripts and Jupyter Notebooks, on experiements with transfer learning and more precisely neural style transfer. I will restrict to the scenario of transforming a photograph to *look like* a famous painting (Van Gogh's *The Starry Night* unless otherwise noted). The interest is at once practical: running times of various algorithms, optimization schemes, etc. *normal/old hardware* and abstract: how pleasant the results are. I will use these two aspects as soft metrics throughout.

**Idea and use case scenario.** Suppose an artist, owning past-its-prime hardware (perhaps a 7 or 10 year-old Macbook Pro/Air, perhaps a 3-4 year-old PC), would like to add neural style transfer to their techniques. Is that possible given the contraints? Is that feasible? Can subjectively interesting results be achieved given enough time? Is the investment in learning neural networks and using a machine learning platform worth it despite the possible impracticality of the actual algorithms? 

**Objectives.** Here are some two important objectives guiding the experiments

- achieve interesting (for some *reasonable definition of interesting*) results on a human scale budget, on middle-to-low-level hardware (could be years old), perhaps without using GPU training or even CPU parallelization
- achieve human-scale *displayable and printable* images; note that there is a large difference between the two image scales here, and my experiments will for the most part use images of **size 533x400** width times height, which is the *display scale*. The *printable scale* is many times bigger: for a 30x40 cm print, one is looking at resolutions of 20+ megapixels, or gigantic resolutions on the order of **6000x4500** pixels. The printable scale is perhaps beyond any reasonable machine learning algorithm for the moment, at least without any serious upsampling or other pre/post processs image manipulation tricks, and this scale is certainly beyond any *human-scale computer* (i.e. one as described in the previor paragraph) can do

## Under construction

## References

- Chollet, *Deep learning with Python*, second edition, [publisher's website](https://www.manning.com/books/deep-learning-with-python-second-edition) and [GitHub repositiory](https://github.com/fchollet/deep-learning-with-python-notebooks)
- Chollet, *Transfer learning and fine-tuning*, [Keras website tutorial](https://keras.io/guides/transfer_learning/)
- Gatys, Ecker, Bethge, *Image style transfer using convolutional neural networks*, [link to PDF](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
- Log0, *Neural style painting*, [GitHub repository](https://github.com/log0/neural-style-painting)
- Narayanan, *Convolutional neural networks for artistic style transfer*, [website](https://harishnarayanan.org/writing/artistic-style-transfer/) and [GitHub repository](https://github.com/titu1994/Neural-Style-Transfer)
