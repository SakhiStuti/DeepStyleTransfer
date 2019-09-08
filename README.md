# DeepStyleTransfer
This repository contains code for style transfer. Baseline trains a network which is fixed for a style image. The improved code trains a network which can take arbitrary content and style image as input.
## Description of the folders:
1. baseline_DATA: Folder contains a small subset of data used for training the baseline model.
	* content : Folder contains the content images used.
		* train : Folder contains the training content images
		* validation : Folder contains the validation content images
		* test : Folder contains the test content images
	* mosaic.jpg: The style image for which the network is trained.
2. baseline_CODE: Folder contains the code for running the training for the baseline model.
	* LOGS : Folder stores the logs and models saved while training
		* tensorboard : Folder contains the tensorboard files generated while training
		* model : Folder contains the model checkpoints generated while training
		* test_image : Saves the output style image for the test images while training
3. DATA : Folder contains a small subset of data used.
	* content : Folder contains the content images used.
		* train : Folder contains the training content images
		* validation : Folder contains the validation content images
  	* style : Folder contains the style images used.
		* train : Folder contains the style train images
		* validation : Folder contains the style validation images

4. CODE : Folder contains the code for running the training, inference as well as GUI.
	* LOGS : Folder stores the logs and models saved while training
		* tensorboard : Folder contains the tensorboard files generated while training
		* model : Folder contains the model checkpoints generated while training
	* models : Folder contains trained decoder model. The encoder model must be downloaded from the drive link provided.
	* RESULTS_gui : Folder stores the images which are saved through the gui
	* RESULTS_inference : Folder stores the images which are saved through inference script
	* SAMPLE_TEST_IMAGES : Folder contains sample content and style images to run the gui and inference
		* content : Folder contains sample content images
		* style : Folder contains sample style images

## Description of the main python scripts:
### baseline_CODE folder:
* train.py

This file uses the following python scripts as dependencies:
* net : contains the class for style network
* vgg : contains the class for vgg network used while evaluating loss
* loss_network : contains the class for the loss network
* helper_functions : contains helper function required by the network

### CODE folder:
* train.py - file which performs the training
* inference - file which performs the inference
* gui - file which runs the gui

The above three files use the following python scipts as dependencies:
* dataloader : defines the class for loading dataset
* network : contains the classes for networks required for training and inference
* encoder : contains the class defining the encoder 
* decoder : contains the class defining the decoder
* helper_function : contains helper function required by the network

## Requirements:
**My code runs on the following versions**
* Pytorch 1.0.1
* TensorboardX 1.6
* tqdm 4.28.1
* PIL 5.3.0
* torchvision 0.2.2
* tkinter

## Command to run:

### Baseline: 
Download the vgg model file from my drive https://drive.google.com/file/d/1YiqXQpfAc_FATeJe2_n_jC_6HsineYVr/view?usp=sharing and place it in the baseline_code folder. Now, you are ready to run the network for training.
1. TRAINING: All the required paths have been set.  
	python train.py   
	      or  
	python train.py --content_dir ** --style_image_path ** --val_dir **  --lr **    
Sample Images for gui and inference can be taken from the SAMPLE_TEST_IMAGES folder. They are test images.
### Improved: 
Download the encoder model file from my drive https://drive.google.com/open?id=1HTBmh4YQa-b-z4Zljra8mJWtJrdItct- and place it in the models folder. Now, you are ready to run the network for training, inference as well as gui.
1. TRAINING: All the required paths have been set.  
	python train.py   
	      or  
	python train.py --content_dir ** --style_dir ** --val_content_dir ** --val_style_dir ** --lr **  
2. INFERENCE: The sample images have been set up in th file.  
	python inference.py  
              or  
	python inference.py --content_path ** --style_path **  
3. GUI : python gui.py  
Sample Images for gui and inference can be taken from the SAMPLE_TEST_IMAGES folder. They are test images.

## Demo Video 
https://youtu.be/gdYi0k3Yp8g

## References
1.  J. Johnson, A. Alahi, and L. Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. 2016. 2, 3
2. Huang, X., Belongie, S.: Arbitrary style transfer in real-time with adaptive instance normalization. In: ICCV. (2017)
3. K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015. 4, 5
4. https://github.com/naoto0804/pytorch-AdaIN
5. https://github.com/pytorch/examples/tree/master/fast_neural_style

