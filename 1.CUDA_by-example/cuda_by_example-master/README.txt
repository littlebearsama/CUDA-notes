--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
CUDA by Example: An Introduction to General-Purpose GPU Programming
README.txt
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
July 2010
Copyright (C) 2010 NVIDIA Corp.



Distribution Contents
----------------------------------------------------
The end user license (license.txt)
Code examples from chapters 3-11 of 
     "CUDA by Example: An Introduction to General-Purpose GPU Programming"
Common code shared across examples
This README file (README.txt)



Compiling the Examples
----------------------------------------------------
The vast majority of these code examples can be compiled quite easily by using 
NVIDIA's CUDA compiler driver, nvcc. To compile a typical example, say 
"example.cu," you will simply need to execute:

> nvcc example.cu

The compilation will produce an executable, a.exe on Windows and a.out on Linux.
To have nvcc produce an output executable with a different name, use the 
-o <output-name> option. To learn about additional nvcc options, run

> nvcc --help



Compiling Examples for Compute Capabilities > 1.0
----------------------------------------------------
The examples from Chapter 9, hist_gpu_gmem_atomics.cu and 
hist_gpu_shmem_atomics.cu, both require GPUs with compute capabilities greater 
than 1.0. Likewise, the examples from Appendix A, dot.cu and hashtable_gpu.cu,
also require a GPU with compute capability greater than 1.0.

Accordingly, these examples also require an additional argument in order to 
compile and run correctly. Since hist_gpu_gmem_atomics.cu requires compute 
capability 1.1 to function properly, the easiest way to compile this example
is,

> nvcc -arch=sm_11 hist_gpu_gmem_atomics.cu


Similarly, hist_gpu_shmem_atomics.cu relies on features of compute capability
1.2, so it can be compiled as follows:

> nvcc -arch=sm_12 hist_gpu_shmem_atomics.cu




Compiling Examples with OpenGL and GLUT Dependencies
----------------------------------------------------

The following examples use OpenGL and GLUT (GL Utility Toolkit) in order to 
display their results:

Chapter 4                       Chapter 7
    julia_cpu.cu                    heat.cu 
    julia_gpu.cu                    heat_2d.cu

Chapter 5                       Chapter 8
    ripple.cu                       basic.cu
    shared_bitmap.cu                basic2.cu
                                    heat.cu
Chapter 6                           ripple.cu
    ray.cu
    ray_noconst.cu


To build with OpenGL and GLUT, some additions will need to be made to the nvcc
command-line. These instructions are different on Linux and Windows operating 
systems.


Linux
-----------------------
On Linux, you will first need to ensure that you have a version of GLUT 
installed. One method for determining whether GLUT is correctly installed is 
simply attempting to build an example that relies on GLUT. To do this, one
needs to add -lglut to the nvcc line, indicating that the example needs to be
linked against libglut. For example:

> nvcc -lglut julia_gpu.cu

If you get an error about missing GL/glut.h or a link error similar to the 
following, GLUT is not properly installed:

    /usr/bin/ld: cannot find -lglut


If you need to install GLUT, we recommend using freeglut on Linux systems. As
always with Linux, there exist a variety of ways to install this package, 
including downloading and building a source package from
http://freeglut.sourceforge.net/

The easiest method involves exploiting the package managers available with many
Linux distributions. Two common methods are given here:

> yum install freeglut-devel

> apt-get install freeglut-dev



Windows
-----------------------
This distribution includes both 32-bit and 64-bit versions of GLUT, pre-built 
for Windows. You are free to ignore these, but using them will be your quickest
method to get up and running.

For example, to compile the heat transfer simulation in Chapter 7, we will need
to explicitly tell nvcc where to find the GLUT library. If we are in the 
directory where we've extracted this distribution, we can add the argument -Llib 
to tell nvcc to look in .\lib for additional libraries. 

> nvcc -Llib chapter07\heat.cu

When we proceed to run the resulting a.exe, we will also need to ensure that 
glut32.dll (on 32-bit Windows) or glut64.dll (on 64-bit Windows) can be found
on our PATH (or that there's a copy in the directory containing a.exe). These
files are located in the bin\ directory of the distribution.

In the Linux-specific instructions, we recommended freeglut. Note that
freeglut is also available for Windows platforms, so you should feel free to 
download and use the Windows freeglut. However, if you choose to do so, the 
rest of these instructions will not be useful.



Windows Notes
-------------

o To compile from the command-line on Windows, it is recommended that you use
  the command-line shortcut installed by Visual Studio. On 64-bit systems with
  non-Express Editions of Visual Studio, this shortcut will be named:
  "Visual Studio <version> x64 Win64 Command Prompt." On 32-bit systems or on 64-bit
  systems with Visual Studio Express Edition, this shortcut will be named,
  "Visual Studio <version> Command Prompt."

o If you are using a 64-bit system with Visual Studio Express Edition, you will
  need an additional command-line argument to nvcc in order to compile 32-bit
  executables. This is a consequence of the Express Edition not containing 64-bit
  compilation tools. Without the -m32 command-line argument, nvcc defaults to 64-bit 
  builds when it detects a 64-bit system (which fails to link because Visual Studio
  Express Edition only contains 32-bit runtime libraries). 

  For example, to compile Chapter 3's "Hello, World!" example:

  > nvcc -m32 hello_world.cu

o Individual kernels are limited to a 2-second runtime by Windows
  Vista and Windows 7. Kernels that run for longer than 2 seconds 
  will trigger the Timeout Detection and Recovery (TDR) mechanism. 
  For more information, see
  http://www.microsoft.com/whdc/device/display/wddm_timeout.mspx.
  
  This issue may specifically be a problem on slower GPUs when running
  the gmem histogram example in Chapter 9 or the GPU hashtable example in
  Appendix A. To work around this issue, try running these examples with a 
  smaller value for SIZE.

