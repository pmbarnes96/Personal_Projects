This folder is for abstract art I created with a GAN.  The folder titled "Generated Gallery" contains 100 randomly sampled images created by the generator.  The files in Generated Gallery titled "Generated Grid" show 25 images at once for more convenient viewing.  The file titled "Training_GAN.py" contains the code used to train this GAN.

The GAN [1] is trained on a set of 2872 images downloaded from [2].  The architure is that of a deep convolutional GAN [3] capable of working with 256x256 RGB images.  To improve the stability of training, a Wasserstein loss is used [4] rather than binary cross entropy.  A gradient penalty [5] is used to discourage violations of 1-Lipschitz continuity in the critic function.

Visual inspection of the generated gallery of images reveals a diverse distribution of abstract pictures, demonstrating a lack of mode collapse.  None of the generated images look anywhere near identical to any images in the training set, demonstrating that the GAN is not overfitting.

Bibliography:

[1] https://arxiv.org/abs/1406.2661

[2] https://www.kaggle.com/datasets/bryanb/abstract-art-gallery

[3] https://arxiv.org/abs/1511.06434

[4] https://arxiv.org/abs/1701.07875

[5] https://arxiv.org/abs/1704.00028
