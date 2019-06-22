This is the Adobe Deep Image Matting Dataset.

This dataset was introduced in the work:
Ning Xu, Brian Price, Scott Cohen, Thomas Huang.  Deep Image Matting.  In Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR) 2017.

Please refer to the included license for information regarding the allowed use of the dataset.


Some important points about the dataset:

1)  We are releasing the foreground images and the alpha mattes along with code to composite the images.  To use the code, simply make sure that the directories at correct at the top of the file.  To use the standard training set, make sure that the background image directory contains the MSCOCO dataset and do not change the other parameters in the code.  To use the test set, make sure the background directory contains the PASCAL VOC.  There is code for the training set and the test set, but the only difference between them is the variable defining the number of background images for each foreground.

2)  We have received concerns that different platforms may order the images differently given our code.  Accordingly, we have included a list of the images in the correct order for the training and test sets.  

Here is an example of how the list works.  For the test set, each test fg is composited onto 20 test bg images.  So in the file test_fg_names.txt, the first entry (16452523375_08591714cf_o.png) should be composited onto the first 20 images listed in test_bg_names.txt, the second entry in test_fg_names.txt should be composited onto the next 20 images in test_bg_names.txt., etc.

3)  Although the original paper said there were 493 unique images used in the training set, we just realized while putting this dataset together that some of the images were double counted and there are only 455 images.
 
4)  We are not able to release 24 of the training images.  This means that the training set will contain 431 objects that when composited on 100 backgrounds will create 43,100 training images.
 
5)  Also, some of the training images were gathered from the internet and are licensed under various Creative Commons licenses.  As such, they are not subject to the Adobe license that we sent.  These are contained in a separate folder.  An accompanying file gives the url of these images where you can see the image license and any attribution information.  All test images are covered by the Adobe license.
 
6)  Please note that some of the images are quite large.  We trained with these large images, but feel free to downsample them if necessary.

7)  For academic integrity, please do not use the test set for training or validation.  Only use it for final testing.
