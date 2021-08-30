# Basic Cloud Renderer in Vulkan



![sample](.\sample.png)



This is a project adapting the vulkan tutorial (https://vulkan-tutorial.com/) to have a simple compute shader for volumetric pathtracing of heterogeneous medium.



The bindings of the shader in Shader/cloud.comp are:

0 - The output image (mean radiance over all passes)

1 - The grid (sampler3D), this image is sampled using stochastic sampling to approximate a linear sampling using nearest sampling.

2 - Uniform Buffer Object with all global parameters: Here is camera (proj2world matrix) and other parameters referring to the cloud and environment lighting.

3 - Uniform Buffer Object with the frame number: This buffer of one integer is updated every "frame" to represent a new pass of the pathtracing. Because the only thing changing between "frames" is this integer, we avoid recreating command buffers, but instead update this constant buffer. Notice ,push constants are not used because they have to be saved in the command buffer.

4 - accumulation image (Store Image), this image is the total sum of all radiances computed.

5 - firstX image (Store Image), this image is the last frame "first scattering position" and the pdf of such position in the w component. (For denoising purposes)

6 - firstW image (Store Image), this image is the last frame "first scattering direction" and the pdf of such direction  in the w component. (For denoising purposes).



In the main.cpp file there are several defines for parameters of the application, scene and cloud properties.