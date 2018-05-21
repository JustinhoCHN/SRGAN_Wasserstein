## SRGAN_Wasserstein
Applying Waseerstein GAN to SRGAN, a GAN based super resolution algorithm.

***This repo was forked from @zsdonghao 's [tensorlayer/srgan](https://github.com/tensorlayer/srgan) repo, based on this original repo, I changed some code to apply wasserstein loss, making the training procedure more stable, thanks @zsdonghao again, for his great reimplementation.***

### SRGAN Architecture
![](http://ormr426d5.bkt.clouddn.com/18-5-18/43943225.jpg)

TensorFlow Implementation of ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802)

### Wasserstein GAN

When the SRGAN was first proposed in 2016, we haven't had [Wasserstein GAN](https://arxiv.org/abs/1701.07875)(2017) yet, WGAN using wasserstein distance to measure the disturibution difference between different data set. As for the original GAN training, we don't know when to stop training the discriminator or the generator, to get a nice result. But when using the wasserstein loss, as the loss decreasing, the result will be better. So we are going to use the WGAN and we are not going to explain the math detail of WGAN here, but to give the following steps to apply WGAN.

* Remove the sigmoid activation from the last layer of the discriminator. (```model.py```, line 218-219)
* Don't take logarithm to the loss of discriminator and generator. (```main.py```, line 105-108)
* Clipping the weights to some contant range [-c, c]. (```main.py```, line 136)
* Don't use the optimizer like adam or momoentum which based on momentum, instead, RMSprop or SGD would be better. (```main.py```, line 132-133)

These above steps was given by an excellent article[[4]](https://zhuanlan.zhihu.com/p/25071913), the arthor explained the WGAN in a very straightforward way, it was written in Chinese.

### Loss curve and Result
![](http://ormr426d5.bkt.clouddn.com/18-5-18/8141442.jpg)

![](http://ormr426d5.bkt.clouddn.com/18-5-18/22508558.jpg)

![](http://ormr426d5.bkt.clouddn.com/18-5-18/83166966.jpg)

![](http://ormr426d5.bkt.clouddn.com/18-5-18/96883821.jpg)


### Prepare Data and Pre-trained VGG

- 1. You need to download the pretrained VGG19 model in [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) as [tutorial_vgg19.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_vgg19.py) show.
- 2. You need to have the high resolution images for training.
  -  In this experiment, I used images from [DIV2K - bicubic downscaling x4 competition](http://www.vision.ee.ethz.ch/ntire17/), so the hyper-paremeters in `config.py` (like number of epochs) are seleted basic on that dataset, if you change a larger dataset you can reduce the number of epochs. 
  -  If you dont want to use DIV2K dataset, you can also use [Yahoo MirFlickr25k](http://press.liacs.nl/mirflickr/mirdownload.html), just simply download it using `train_hr_imgs = tl.files.load_flickr25k_dataset(tag=None)` in `main.py`. 
  -  If you want to use your own images, you can set the path to your image folder via `config.TRAIN.hr_img_path` in `config.py`.

### Run

We run this script under [TensorFlow](https://www.tensorflow.org) 1.4 and the [TensorLayer](https://github.com/tensorlayer/tensorlayer) 1.8.0+.

* Installation

```
pip install tensorlayer==1.8.0
conda install tensorflow-gpu==1.3.0
pip install tensorflow-gpu==1.4.0
pip install easydict
```

-  You can download [DIV2K - bicubic downscaling x4 competition](http://www.vision.ee.ethz.ch/ntire17/) dataset, and set your image folder in `config.py`. 
- Other links for DIV2K, in case you can't find it : [test\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_bicubic_X4.zip), [train_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip), [train\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip), [valid_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_valid_HR.zip), [valid\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip).

```python
config.TRAIN.img_path = "your_image_folder/"
```
- Tenserboard logdir.

I added the tensorboard callbacks to monitor the training procedure, please change the logdir to your folder.

```python
config.VALID.logdir = 'your_tensorboard_folder'
```

- Start training.

```bash
python main.py
```

- Start evaluation. ([pretrained model](https://github.com/tensorlayer/srgan/releases/tag/1.2.0) for DIV2K) 
**An important note:**
This pretrained weights is provided by the original author @zsdonghao , his final layer's conv kernel of ```SRGAN_g``` (model.py line 53) is using 1×1 kernel, but I changed this kernel to 9×9, so if you use this pretrained weights, you may get the weights unequal error.
Two advice:
1)Train the whole network from scratch, you'll get the 9×9 version weights, for further training or evaluating images.
2)You can just change the ```SRGAN_g``` 's final conv kernel (```model.py``` line 53) to (1, 1) instead of (9, 9), and change the ```model.py``` line 35 conv kernel from (9, 9) to (3, 3), so that you can use the pretrained weights.

```bash
python main.py --mode=evaluate 
```

### What's new?

Compare with the original version, I did the following changes:

1. Adding WGAN, as described in Wasserstein GAN chapter.
2. Adding tensorboard, to monitor the training procedure.
3. Modified the last conv layer of 'SRGAN_g' in ```model.py``` (line 100), changing the kernel size from (1, 1) to (9, 9), as the paper proposed.

### Reference
* [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
* [2] [Is the deconvolution layer the same as a convolutional layer ?](https://arxiv.org/abs/1609.07009)
* [3] [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
* [4] [令人拍案叫绝的Wasserstein GAN](https://zhuanlan.zhihu.com/p/25071913)
* [5] [SRGAN With WGAN，让超分辨率算法训练更稳定-知乎专栏](https://zhuanlan.zhihu.com/p/37009085) [Chinese verson readme]

### Author
- [zsdonghao](https://github.com/zsdonghao)
- [justinho](https://github.com/JustinhoCHN)

### License

- For academic and non-commercial use only.
- For commercial use, please contact tensorlayer@gmail.com.