**Kernels**
  1. Kernel can be called a feature extractor.
  2. In order for a network or a system to be able to classify images or image processing tasks, it is important that the system understands various parts and features of image. Like it should be able to recognise different objects, to do that it needs to know parts of objects, textures and patterns, edges and gradients.
  3. Extraction of these features from images can be done by using a filter that is convolved on pixel values of images. These filters are called kernels.
  4. Kernels are basically n*n matrices, like below
  ![alt text](http://www.davidsbatista.net/assets/images/2018-03-31_dpln_0412_cnn.png)
  5.Always choose odd number of kernels, even numbers have no middle point, hence it becomes tough to use the available matrix values to distribute filter values across left and right. This will lead to wasting a lot pixel values.
  6. **We almost always use 3*3 matrix for Kernels**.
  
    i. GPU's are well optimized for 3*3 matrices.
    
    ii. Using a 3*3 kernel in contrast with higher dimension kernels reduces number of parameters required.
    
    Eg: IMAGE(size=5*5) > Kernel(size=5*5)(25 parameters+1 bias) > Output (size = 1*1)-----> Total parameters 26
    
        IMAGE(size=5*5) > Kernel(size=3*3)(9 parameters + 1 bias) >Output1(size 3*3) > Kernel(size=3*3)(9 + 1 parameters) > Output(size= 1*1)--------> Total parameters required => 10+10 = 20
        
        As you see two 3*3 kernels were required to arrive at same output of 1*1 on the same image, where as a single 5*5 kernel did the job. 
        But 5*5 kernel channel used 25 parameters, where as two 3*3 kernels use just 18 parameters. Hence using 3*3 will help tune our model with very less parameters.
        
  
 **Channels**

  1. If we can divide our image into several parts such that, when these parts are combined in a some combination it yields the original image, then these individual parts are called Channels.
  2. A channel is expected to contain similar information
  3. Color images can be easily broken down to R(red) G(green) B (blue) channels. Upon combining these, we get the original image.
  4. LCD screens use RGB channels.
  5. Newspapers and prints use CMYK channels.Cyan, Magenta, Yellow, and Black to display the same images.
  
  
  **How many times a 3*3 kernel should be used to get to 1*1 image size from 199* 199**
  --> Running 1 3*3 kernel reduces image size by n-2. So nearly about (199-1)/2 = 98 kernels need to be used. Below is calculations:
  199 | 197 | 195 | 193 | 191 | 189 | 187 | 185 |183 | 181 | 179 | 177 | 175 | 173 | 171 | 169 | 167 | 165 | 163 | 161 | 159 | 157 | 155 | 153 | 151 | 149 | 147 | 145 | 143 | 141 | 139 | 137 | 135 | 133 | 131 | 129 | 127 | 125 | 123 | 121 | 119 | 117 | 115 | 113 | 111 | 109 | 107 | 105 | 103 | 101 | 99 | 97 | 95 | 93 | 91 | 89 | 87 | 85 | 83 | 81 | 79 | 77 | 75 | 73 | 71 | 69 | 67 | 65 | 63 | 61 | 59 | 57 | 55 | 53 | 51 | 49 | 47 | 45 | 43 | 41 | 39 | 37 | 35 | 33 | 31 | 29 | 27 | 25 | 23 | 21 | 19 | 17 | 15 | 13 | 11 | 9 | 5 | 3 | 1
