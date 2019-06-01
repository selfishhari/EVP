**Channels**

  1. If we can divide our image into several parts such that, when these parts are combined in a some combination-yields original image, then these individual parts are called Channels.
  2. A channel is expected to contain similar information
  3. Color images can be easily broken down to R(red) G(green) B (blue) channels. Upon combining these, we get the original image.
  4. LCD screens use RGB channels.
  5. Newspapers and prints use CMYK channels.Cyan, Magenta, Yellow, and Black to display the same images.
  
**Kernels**
  1. Kernel can be called a feature extractor.
  2. In order for a network or a system to be able to classify images or image processing tasks, it is important that the system understands various parts and features of image. Like it should be able to recognise different objects, to do that it needs to know parts of objects, textures and patterns, edges and gradients.
  3. Extraction of these features from images can be done by using a filter that is convolved on pixel values of images. These filters are called kernels.
  4. Kernels are basically n*n matrices, like below
![alt text] (http://www.davidsbatista.net/assets/images/2018-03-31_dpln_0412_cnn.png)
