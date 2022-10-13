# Let-s-play-GANs
### Objective
Generate synthetic dice pictures by different types of GANs//
In this repo, I implemented mixture of GANs and conditional GANs.

### Introduction
Files:
1. dataset.py: rewrite the file that TA attached
2. evaluator.py: the pretrained model for computing the acore
3. main.py:test test.json and show the score and save the image
4. best_eval_test.png:best image
5. acgan+dcgan.py(ipynb):acgan+dcgan, but I do a little change in the dcgan
6. cgan+dcgan.py(ipynb):cgan+dcgan
7. cgan+dcgan+wgan.py:try different loss function, but performance is bad
8. cgan+srgan.py:failed, I don’t know where is the mistake..it generate some weird images
9. new_acgan+dcgan.py: change the loss function based on architecture:acgan+dcgan

Folders:
1. cgan_srgan: some bad results. Perhapes some logic errors.
2. cgan_dcgan_wgan: results are not promising.
3. cgan_dcgan: results are not promising.
4. acgan_dcgan: Convincing result with auxiliary loss function. (Final submission to TA)
