# 目录



[TOC]


# 一些知识

## 缓冲区(buffer)与缓存区(cache)

### **一、缓冲**

缓冲区(buffer)，它是内存空间的一部分。也就是说，在内存空间中预留了一定的存储空间，这些存储空间用来缓冲输入或输出的数据，这部分预留的空间就叫做缓冲区，显然缓冲区是具有一定大小的。

缓冲区根据其对应的是输入设备还是输出设备，分为输入缓冲区和输出缓冲区。

### 二、缓存

1. CPU的Cache，它中文名称是高速缓冲存储器，读写速度很快，几乎与CPU一样。由于CPU的运算速度太快，内存的数据存取速度无法跟上CPU的速度，所以在cpu与内存间设置了cache为cpu的数据快取区。当计算机执行程序时，数据与地址管理部件会预测可能要用到的数据和指令，并将这些数据和指令预先从内存中读出送到Cache。一旦需要时，先检查Cache，若有就从Cache中读取，若无再访问内存，现在的CPU还有一级cache，二级cache。简单来说，Cache就是用来解决CPU与内存之间速度不匹配的问题，避免内存与辅助内存频繁存取数据，这样就提高了系统的执行效率。
2. 磁盘也有cache,硬盘的cache作用就类似于CPU的cache，它解决了总线接口的高速需求和读写硬盘的矛盾以及对某些扇区的反复读取。

### **三、缓存（cache）与缓冲(buffer)的主要区别**

Buffer的核心作用是用来缓冲，缓和冲击。比如你每秒要写100次硬盘，对系统冲击很大，浪费了大量时间在忙着处理开始写和结束写这两件事嘛。用个buffer暂存起来，变成每10秒写一次硬盘，对系统的冲击就很小，写入效率高了，日子过得爽了。极大缓和了冲击。

Cache的核心作用是加快取用的速度。比如你一个很复杂的计算做完了，下次还要用结果，就把结果放手边一个好拿的地方存着，下次不用再算了。加快了数据取用的速度。

简单来说就是buffer偏重于写，而cache偏重于读。



# 第三章 简介

* 将CPU即系统的内存称为主机（**host**），而将GPU及其内存称为设备（**device**）

```C++
#include<stdio.h>

__global__ void add(int a,int b,int *c){
  *c = a + b;
}
int main(){
  int c;
  int *dev_c;
  cudaMalloc((void**)&dev_c,sizeof(int));
  add<<<1,1>>>(2,7,dev_c);
  cudaMemcpy(&c,
             dev_c,
             sizeof(int),
             cudaMemcpyDeviceToHost);
  printf("2 + 7 = %d",c);
  cudafree（dev_c）;
  return 0;
}

```

## 1.核函数调用

- 1.函数的定义带有了**`__global__`**这个标签，表示这个函数是在GPU上运行。函数add()将被交给**“编译设备代码的编译器”**。需要指出的是尽管是在GPU上执行，但是仍然是由**CPU端发起调用的**。
- 在每个启动线程中都被调用一遍。


- 2.主机代码发送给一个编译器，而将设备代码发送给另外一个编译器（**CUDA编译器**）,CUDA编译器运行时**将负责实现从主机代码中调用设备代码**。
- 3.核函数相对于CPU代码是异步的，也就是控制会在核函数执行完成之前就返回，这样**CPU就可以不用等待核函数的完成而继续执行后面的CPU代码**
- 4.核函数内部只能访问**device内存**。因为核函数是执行在设备端，所以只能访问设备端内存。所以要使用**cudaMalloc**在**GPU**的内存(全局内存)里开辟一片空间。用来存放结果*dev_c。再通过**cudaMemcpy**这个函数把内容**从GPU**复制出来。

函数部分前缀：

| 限定符                       | 在哪里被调用                | 在哪里被执行 |
| -------------------------- | --------------------------- | ------------ |
| **`__host__`**（默认缺省） | 仅由CPU调用                 | 由CPU执行    |
| **`__gobal__`**            | 仅由CPU调用                 | 由GPU执行    |
| **`__device__`**           | 仅由GPU中一个线程调用的函数 | 由GPU执行    |

限制：

1. **`__host__`**：  

   限定符无法一起使用 

2. **`__gobal__`**：

   限定符无法一起使用；

   函数不支持递归；

   函数的函数体内无法声明静态变量；

   函数不得有数量可变的参数；

   支持函数指针；

   函数的返回类型必须为空；

   函数的调用是异步的，也就是说它会在设备执行完成之前返回；

   函数参数将同时通过共享存储器传递给设备，且限制为 256 字节；

3. **`__device__`**：

   函数不支持递归；

   函数的函数体内无法声明静态变量；

   函数不得有数量可变的参数；

   函数的地址无法获取

4. 之前说了**`__host__`**和**`__gobal__`**限定符无法和其他限定符使用，但与 **`__device__`**限定符不是

**`__constant__`** 限定符可选择与 **`__device__`**限定符一起使用，所声明的变量具有以下特征：

1.位于固定存储器空间中；2. 与应用程序具有相同的生命周期；3.可通过网格内的所有线程访问，也可通过运行时库从主机访问。

**`__shared__`** 限定符可选择与 **`__device__`**限定符一起使用，所声明的变量具有以下特征：1.位于线程块的共享存储器空间中；2. 与块具有相同的生命周期；3.尽可通过块内的所有线程访问。只有在` _syncthreads()`_的执行写入之后，才能保证共享变量对其他线程可见。除非变量被声明为瞬时变量，否则只要之前的语句完成，编译器即可随意优化共享存储器的读写操作。

##  2.参数传递

* `<<<>>>`尖括号表示要将一些参数传递给运行时系统，**这些参数并不是传递给设备代码的参数**，而是告诉运行时**如何启动设备代码**。传递给设备代码本身的参数是放在圆括号中传递的。

  > 尖括号作用？**线程配置**。
  >
  > <<<Dg, Db, Ns, S>>>
  >
  > 1. Dg 的类型为 dim3，指定网格的维度和大小，**Dg.x * Dg.y 等于所启动的块数量**，Dg.z =1无用，目前还不支持三维的线程格；如果指定Dg=256，那么将有256个线程块在GPU上运行。
  > 2. Db 的类型为 dim3，指定各块的维度和大小，Db.x * Db.y * Db.z **等于各块的线程数量**；
  > 3. Ns 的类型为 size_t，指定各块为此调用动态分配的共享存储器（除静态分配的存储器之外），这些动态分配的存储器可供声明为外部数组的其他任何变量使用，Ns 是一个可选参数，默认值为 0；
  > 4. S 的类型为 cudaStream_t，指定相关流；S 是一个可选参数，默认值为 0。


* 核函数内部可以调用CUDA**内置变量**，比如threadIdx，blockDim等。下下章将具体谈到线程索引。
* 参数传递和普通函数一样，**通过括号内的形参传递。**


# 第四章 CUDA C 并行编程

```C++
#include<stdio.h>

#define N   10

__global__ void add( int *a, int *b, int *c ) {
    int tid = blockIdx.x;    // this thread handles the data at its thread id
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main( void ) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a, N * sizeof(int) );
    cudaMalloc( (void**)&dev_b, N * sizeof(int) );
    cudaMalloc( (void**)&dev_c, N * sizeof(int) );

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice );

    add<<<N,1>>>( dev_a, dev_b, dev_c );

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy( c, dev_c, N * sizeof(int),
                              cudaMemcpyDeviceToHost );

    // display the results
    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }

    // free the memory allocated on the GPU
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );
    return 0;
}
```

* 调用cudaMalloc()在**设备上**为三个数组分配内存。
* 使用完GPU后调用cudaFree()来释放他们。
* 通过cudaMemcpy()进行主机与设备之间复制数据。




# 第五章 线程协作

  ## 1.GPU逻辑结构

- CUDA的**软件架构**由网格（Grid）、线程块（Block）和线程（Thread）组成，

  相当于把GPU上的计算单元分为若干（2~3）个网格，

  每个网格内包含若干（65535）个线程块，

  每个线程块包含若干（512）个线程，

  三者的关系如下图：

  ![img](https://img-blog.csdn.net/20170204230954920)

## 

## 2.线程索引（ID）定位

**作用：**

线程ID用来定位线程，根据线程ID来给各个线程分配数据以及其他操作。

计算线程ID需要通过本线程的各个内建变量来计算被调用核函数所进入的线程ID.

**内建变量：**

1. **threadIdx(.x/.y/.z代表几维索引)**：线程所在block中各个维度上的线程号

2. **lockIdx(.x/.y/.z代表几维索引)**：块所在grid中各个维度上的块号

3. **blockDim(.x/.y/.z代表各维度上block的大小)**：

   block的大小即block中线程的数量，

   blockDim.x代表块中x轴上的线程数量，

   blockDim.y代表块中y轴上的线程数量，

   blockDim.z代表块中z轴上的线程数量

4. **gridDim(.x/.y/.z代表个维度上grid的大小)**：

   grid的大小即grid中block的数量，

   gridDim.x代表grid中x轴上块的数量，

   gridDim.y代表grid中y轴上块的数量，

   gridDim.z代表grid中z轴上块的数量

  **定义grid、block大小：**
  dim3 numBlock(m,n)
  dim3 threadPerBlock(i,j)
  则blockDim.x=i; blockDim.y=j; gridDim.x=m; gridDim.y=n

  **kernel调用：**
  kernel<<<numBlock,threadPerBlock>>>(a,b)
  这是调用kernel时的参数，尖括号<<<>>>中第一个参数代表启动的线程块的数量，第二个参数代表每个线程块中线程的数量.

  **总的线程号：**
  设线程号为tid,以下讨论几种调用情况下的tid的值，这里只讨论一维／二维的情况

  **一维：**
  １．kernel<<<1,N>>>()
  block和thread都是一维的，启动一个block，里面有N个thread，１维的。**tid=threadIdx.x**

  ２．kernel<<<N,1>>>()
  启动N个一维的block，每个block里面１个thread。**tid=blockIdx.x**

  ３．kernel<<<M,N>>>()
  启动Ｍ个一维的block，每个block里面N个一维的thread。**tid=threadIdx.x+blockIdx.x * blockDim.x**

### 一般如何配置线程？

1. kernel<<<M,N>>>() M，N为1维度

   输入数据numbers，设定每个线程块有N=128或256或512个线程，一般设为128。

   计算应该设置的线程块M=（numbers+N-1）/N，向上取整；线程块是数量不能超过65535，这是一种硬件限制，如果启动的线程块数量超过了这一限值，那么程序将运行失败。

  **二维：**
  ４．dim grid(m,n)
  kernel<<<grid,1>>>()
  启动一个二维的m*n个block，每个block里面一个thread。**tid=blockIdx.x+blockIdx.y * gridDimx.x**

  ５．dim grid(m,n)
  kernel<<<grid,N>>>()
  启动一个二维的m*n大小的block，每个block里面Ｎ个thread。**tid=**

  ６．dim block(m,n)
  kernel<<<1,block>>>()  
  **tid=**

  ７．dim block(m,n)
  kernel<<<N,block>>>()
  **tid=**

  ８．dim　grid(m,n)
  dim block(i,j)
  kernel<<<grid,block>>>()
  **tid=**

### tid<N

公司M=（numbers+N-1）/N保证了启动了足够多的线程,当输入数据numbers不是线程块里面线程数N的整数倍时,将启动过多线程。

**然而,在核函数中已经解决了这个问题。在访问输入数组和输出数组之前，必须检查线程的偏移（索引）tid是否位于0到N之间**

```C++
if(tid<N)
	c[tid]=a[tid]+b[tid];
```

**因此，当索引越过数组边界时**，例如当启动并行线程数量不是线程块中线程的数目N（128）就会出现这种情况，那么核函数将自动停止执行计算。**更重要的是，核函数不会对越过数组边界的内存进行读取或写入。**

* 简单来说就是启动了充足的线程，而有的线程不用工作，为了防止核函数不会出现越界读取等错误，我们使用了条件判断if（tid<N）。

### 当数据大于运行线程时

因为数据数目大于线程数目，所以正在运行的所有线程都可能会再被执行，直到所有数据处理完毕。所以`if(tid<N)`或`while(tid<N)`不仅仅用于判断线程ID tid，是否执行线程。也用于循环。

添加语句`tid+=blockDim.x*gridDim.x;`增加的值等于**每个线程块中的线程数量乘以线程网格中线程块的数量**，在上面的线程分配(一维的线程格，一维的线程块)中为`blockDim.x*gridDim.x`

故核函数被改写为

```C++
__global__ void add( int *a, int *b, int *c ) {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;    // this thread handles the data at its thread id
    while (tid < N)
        c[tid] = a[tid] + b[tid];
    	tid+=blockDim.x*gridDim.x;//新增的
}
```

## 3.二维的线程格，二维的线程块（实现波纹效果）

![](https://img-blog.csdn.net/20170716092839317?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvRmlzaFNlZWtlcg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

```C++


__global__ void kernel( unsigned char *ptr, int ticks ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // now calculate the value at that position
    float fx = x - DIM/2;
    float fy = y - DIM/2;
    float d = sqrtf( fx * fx + fy * fy );
    unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                         cos(d/10.0f - ticks/7.0f) /
                                         (d/10.0f + 1.0f));    
    ptr[offset*4 + 0] = grey;
    ptr[offset*4 + 1] = grey;
    ptr[offset*4 + 2] = grey;
    ptr[offset*4 + 3] = 255;
}


dim3    blocks(DIM/16,DIM/16);
dim3    threads(16,16);
kernel<<<blocks,threads>>>( data.dev_bitmap, ticks );



```
* blocks和threads是两个二维变量
* 由于生成的是一幅图像，因此使用二维索引，并且每个线程都有唯一的索引`(x,y)`，这样可以很容易与输出图像中的像素一一对应起来。就是输出图像的像素索引`(x,y)`
* offset是数据的线程索引（被称为全局偏置）,该线程对应图像像素索引(x,y)也对应数据索引offset
* `(fx,fy)`=`(x,y)`相对于图像中心点（DIM/2，DIM/2）位置
* 加入线程块是一个16X16的线程数组，图像有DIMXDIM个像素，那么就需要启动DIM/16 x DIM/16个线程块，从而使每一个像素对应一个线程。
* GPU优势在于处理图像时比如1920X1080需要创建200万个线程，CPU无法完成这样的工作。


## 4.共享内存和同步

* 共享内存术语Shared Memory，是位于SM（流多处理器）中的特殊存储器。还记得SM吗，就是流多处理器，大核是也。
* 将关键字**`__share__`**添加到变量声明中，这将是这个变量**驻留**在**共享内存**中。
* block与block的线程无法通信
* 共享内存缓存区驻留在物理GPU上，而不是驻留在GPU以外的系统内存中。因此，**在访问共享内存时的延迟要远远低于访问普通缓存区的延迟**，使得共享内存像**每个线程块的高速缓存**或中间结果暂存器那样高效。
* **想要在线程之间通信，那么还需要一种机制来实现线程之间的同步**，例如，如果**线程A**将一个值写入到共享内存，并且我们希望**线程B**对这个值进行一些操作，那么只有当线程A写入操作完成后，线程B才能开始执行它的操作。**如果没有同步，那么将发生竞态条件**。

这里的例子是点积的例子，就是：
![点积](https://img-blog.csdn.net/20170716094912097?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvRmlzaFNlZWtlcg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

代码：

```C++
__global__ void dot(float *a, float *b, float *c) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;//全局偏移用来索引数据
    int cacheIndex = threadIdx.x;                   //共享内存缓存中的偏移就等于线程索引
    float temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];                    //线程被执行的次数是未知的，数据最终被保存成temp并                                         
        tid += blockDim.x * gridDim.x;              //存入到threadsPerBlock维的cache中
    }
    cache[cacheIndex] = temp;
    __syncthreads();
    //规约运算
    int i = blockDim.x / 2;//取threadsPerBlock的一半作为i值
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        
        __syncthreads();
        i /= 2;
    }
    //结束while()循环后，每个线程块都得到一个值。这个值位于cache[]的第一个元素中，并且就等于该线程中两两元素乘积的加和。然后，我们将这个值保存到全局内存并结束核函数。
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}
```



1. `__shared__` float cache[threadsPerBlock];在共享内存中申请浮点数组，数组大小和线程块中线程数目相同**每个线程块都拥有该共享内存的私有副本。**
2. 共享内存缓存区cache将保存该block内每个线程计算的加和值。
3. **`__syncthreads();`**等待线程块里面的所有线程执行完毕，简称线程同步。确保线程块中的每个线程都执行完**`__syncthreads();`**前面的语句后才会执行下一语句。
4. 用规约运算，我们取threadsPerBlock的一半作为i值，只有索引小于这个值的线程才会执行。只有当线程索引小于i时，才可以把cache[]的两个数据项相加起来。**`__syncthreads()`**作用如下图（下图中是等待4个线程中的相加操作完成）。

​                            **假设cache[]中有8个元素，因此i的值为4。规约运算的其中一个步骤如下图所示**

![加法](https://img-blog.csdn.net/20170716095149539?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvRmlzaFNlZWtlcg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

5. **由于线程块之间无法通信**。只能将每个线程块算出来的值存出来,存到数组c中，最后会返回block数量个c，然后由cpu执行最后的加法。

* 当某些线程需要执行一条指令，而其他线程不需要执行时，**这种情况就称为线程发散（Thread Divergence）**。在正常环境中，发散的分支只会使得某些线程处于空闲状态，而其他线程将执行分支中的代码。但是在**`__syncthreads()`**情况中，线程发散造成的结果有些糟糕。CUDA架将确保，除非线程块中的每个线程都执行了**`__syncthreads()`**，否则没有任何线程能执行**`__syncthreads()`**之后的指令。如果**`__syncthreads()`**位于发散分支中，一些线程将永远无法执行**`__syncthreads()`**。硬件将一直保持等待。
* 下面代码将使处理器挂起，因为GPU在等待某个永远无法发生的事件。

```C++
        if (cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
          __syncthreads();
          }
```



* 例子2（二维线程布置）基于共享内存的位图（略）

# 第六章 常量内存与事件

## 0.光线追踪

* 常量内存用于保存在核函数执行期间不会发生变化的数据。Nvidia硬件提供了64KB的常量内存，并且对常量内存采取了不同于标准全局内存的处理方式。在某些情况中，用常量内存来替换全局内存能有效地减少内存带宽。
* 在光线跟踪的例子中，没有利用常量内存的代码运行时间为1.8ms，利用了常量内存的代码运行时间为0.8ms
* 将球面数组存入常量内存中。

```C++
#include "cuda.h"
#include "../common/book.h"
#include "../common/image.h"

#define DIM 1024

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

struct Sphere {
    float   r,b,g;
    float   radius;
    float   x,y,z;//球心相对于图像中心的坐标
    __device__ float hit( float ox, float oy, float *n ) {
        float dx = ox - x;
        float dy = oy - y;
        //计算来自于(ox,oy)处像素的光线（垂直于图像平面），计算光线是否与球面相交
        //然后计算相机到光线命中球面出的距离
        if (dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            *n = dz / sqrtf( radius * radius );
            return dz + z;
        }
        return -INF;
    }
};
#define SPHERES 20

__constant__ Sphere s[SPHERES];

__global__ void kernel( unsigned char *ptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - DIM/2);//(ox, oy)=(x, y)相对于图像中心点（DIM / 2，DIM / 2）位置或者说将图像移到中心
    float   oy = (y - DIM/2);

    float   r=0, g=0, b=0;
    float   maxz = -INF;
    for(int i=0; i<SPHERES; i++) {
        float   n;
        float   t = s[i].hit( ox, oy, &n );
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    } 

    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    DataBlock   data;
    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    IMAGE bitmap( DIM, DIM );
    unsigned char   *dev_bitmap;

    // 在GPU上分配内存以计算输出位图
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap,
                              bitmap.image_size() ) );

    // 分配临时内存，对其初始化，并复制到GPU上的常量内存，然后释放临时内存
    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
    for (int i=0; i<SPHERES; i++) {
        temp_s[i].r = rnd( 1.0f );
        temp_s[i].g = rnd( 1.0f );
        temp_s[i].b = rnd( 1.0f );
        temp_s[i].x = rnd( 1000.0f ) - 500;
        temp_s[i].y = rnd( 1000.0f ) - 500;
        temp_s[i].z = rnd( 1000.0f ) - 500;
        temp_s[i].radius = rnd( 100.0f ) + 20;
    }
    HANDLE_ERROR( cudaMemcpyToSymbol( s, temp_s, 
                                sizeof(Sphere) * SPHERES) );
    free( temp_s );

    // generate a bitmap from our sphere data
    dim3    grids(DIM/16,DIM/16);
    dim3    threads(16,16);
    kernel<<<grids,threads>>>( dev_bitmap );

    // copy our bitmap back from the GPU for display
    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );

    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );

    HANDLE_ERROR( cudaFree( dev_bitmap ) );

    // display
    bitmap.show_image();
}
```
* 变量前面加上**`__constant__`**修饰符：`__constant__ Sphere s[SPHERES];`常量内存为静态分配空间，所以不需要调用 cudaMalloc(), cudaFree()；
* 在主机端分配临时内存，对其初始化`Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );`在把变量复制到常量内存后释放内存`free( temp_s );`
* 使用函数**`cudaMemcpyToSymbol()`**将变量从主机内存复制到GPU上的常量内存。（cudaMencpyHostToDevice()的cudaMemcpy()之间的唯一差异在于，**`cudaMemcpyToSymbol()`**会复制到常量内存，而cudaMemcpy()会复制到**全局内存**）

## 1.常量内存带来的性能提升

与从全局内存中读取数据相比，从常量内存中读取相同的数据可以节约带宽，原因有二：

1. 对常量内存的单次读操作可以广播到其他的“邻近”线程，这将节约15次读取操作。
2. 常量内存的数据将缓存(cache)起来，因此对相同地址的连续读操作将不会产生额外的内存通信量。

## 2.线程束Warp

在CUDA架构中，线程束是指一个包含32个线程的集合，这个线程集合被“编织在一起”并且以“步调一致（Lockstep）”的形式执行。在程序中的每一行，线程束中的每个线程都将在不同数据上执行相同的指令。

### 线程束

当处理常量内存是，NVIDIA硬件将把**单次内存读取操作**广播到**每半个线程束（Half-Warp）**。在半线程束中包含了16和线程。如果在半线程束中的每个线程都**从常量内存的相同地址上读取数据**，那么GPU只会产生**一次读取请求**并在随后**将数据广播到每个线程**。如果从常量内存中读取大量的数据，那么这种方式生产的内存流量只是全局内存的1/16（大约6%）。

### 常量内存与缓存

但在读取常量内存是，所节约的并不只限于减少94%的带宽。由于这块内存的内容是不会发生变化的，**因此硬件将主动把这个常量数据缓存在GPU上。**在第一次从常量内存的某个地址上读取后，当其他半线程束请求同一地址是，那么将**命中缓存(cahce)**，这同样减少了额外的内存流量。在光线追踪程序中，将球面数据保存在常量内存后，硬件只需要请求这个数据一次。**在缓存数据后，其他每个线程将不会产生内存流量，原因有两个：**1. 线程将在半线程结束的广播中收到这个数据。 2. 从常量内存缓存中收到数据。

### 负面影响

当使用常量内存是，也可能对性能产生负面影响。**半线程束广播功能实际是把双刃剑**。虽然当所有16个线程地址都读取相同地址是，这个功能可以极大地提高性能，但当所有16个线程分别读取不同地址时，它实际上会降低性能。

## 3.使用事件来测试性能

代码：

```C++
    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    // 在GPU上执行一些工作

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime,start, stop );
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
```

* 运行记录事件start时，还指定了第二个参数。` cudaEventRecord( start, 0 );`在上面代码中为0，流(Stream)的编号。
* 当且仅当GPU完成了之间的工作并且记录了stop事件后，才能安全地读取stop时间值。幸运的是，**还有一种方式告诉CPU在某个事件上同步，这个时间API函数就是`cudaEventSynchronize();`,**当`cudaEventSynchronize`返回时，我们知道stop事件之前的所有GPU工作已经完成了，因此可以安全地读取在stop保存的时间戳。
* 由于CUDA事件是直接在GPU上实现的，因此它们不适用于对同时包含设备代码和主机代码的混合代码计时。也就是说，**你通过CUDA事件对核函数和设备内存复制之外的代码进行计时，将得到不可靠的结果**。



# 第七章 纹理内存

* **纹理内存(Texture Memory)**和常量内存一样，纹理内存是另外一种类型的**只读内存**，在特定的访问模式中，纹理内存同样能够提升性能并减少内存流量。
* 虽然纹理内存最初是针对传统的图形处理应用程序而设计的，但在某些GPU计算应用程序中同样非常有用。
* 与常量内存类似的是，**纹理内存同样缓存在芯片上（利用了芯片上的缓存加速）！！！**，因此在某些情况中，它能够减少对内存的请求并提供更高效的内存带宽。
* **纹理缓存**是专门为那些**在内存访问模式中存在大量空间局部性（Spatial Locality）**的图形应用程序而设计的。在某个计算应用程序中，这意味着一个线程**读取的位置**可能与**邻近**的线程的**读取位置**非常接近

![img](https://images2015.cnblogs.com/blog/986608/201702/986608-20170223153922116-466652140.png)

上图中，从数学角度来看，图中的四个地址并非连续的，在一般的CPU缓存模式中，这些地址将不会缓存。但由于GPU纹理内存是专门为了加速这种访问模式而设计的，因此如果在这种情况中使用纹理内存而不是全局内存，那么将获得性能提升。

## 使用纹理内存实现热传导模拟

### 1.算法描述：

1. 环境是一个矩形网格，在网格中随机散布一些”热源“，热源有着不同的固定温度（该点处的温度不会变）
2. 在随时间递进的每个步骤中，我们假设热量在某个单元机器邻接单元之间”流动“/如果某个单元的温度比邻接单元的温度更高，那么热量将从邻接单元传导到该单元。
3. 我们对新单元中心温度的计算方法为，将单元与邻接单元的温差相加起来，加上原有温度：

![1557992811022](C:\Users\xiaoxiong\AppData\Roaming\Typora\typora-user-images\1557992811022.png)

4. 常量k表示模拟过程中热量的流动速率。k值越大，表示系统会更快地达到稳定温度，而k值越小，则温度梯度将存在更长时间。
5. 只考虑上下左右四个邻域的话讲上述式子展开有

![1557993195547](C:\Users\xiaoxiong\AppData\Roaming\Typora\typora-user-images\1557993195547.png)



### 2.实现流程：

1. 给定一个包含**初始输入温度的网格**，将其中作为热源的单元温度值复制到网格的相应单元中。这将覆盖这些单元之前计算出的温度，因此也就确保了”加热单元将保持恒温“这个条件。用下面代码中的`copy_const_kernel()`实现；
2. 给定一个**输入网格**，用上面公式计算出**输出网格**。用下面代码中的`blend_kernel()`实现；
3. 将输入网格和输出网格交换，为下一个计算步骤做好准备。当模拟下一个时间步时，在步骤2中计算得到的输出温度网格将成为步骤1中的输入温度网格。

###  3.代码：（使用的是二维纹理内存）

```C++
#include "cuda.h"
#include "../common/book.h"
#include "../common/image.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED   0.25f

// these exist on the GPU side
texture<float,2>  texConstSrc;
texture<float,2>  texIn;
texture<float,2>  texOut;

__global__ void blend_kernel( float *dst,
                              bool dstOut ) {
    // map from threadIdx/BlockIdx to pixel position
    //线程布置是二维线程格，二维线程块时的像素坐标索引，以及数据偏置
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float   t, l, c, r, b;
    //根据dstOut标志来看读取的输出部分的内存还是输出部分的内存
    if (dstOut) {              
        t = tex2D(texIn,x,y-1);
        l = tex2D(texIn,x-1,y);
        c = tex2D(texIn,x,y);
        r = tex2D(texIn,x+1,y);
        b = tex2D(texIn,x,y+1);
    } else {
        t = tex2D(texOut,x,y-1);
        l = tex2D(texOut,x-1,y);
        c = tex2D(texOut,x,y);
        r = tex2D(texOut,x+1,y);
        b = tex2D(texOut,x,y+1);
    }
    dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

__global__ void copy_const_kernel( float *iptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;//将线程中的内部线程索引变量变成图像坐标
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;//计算偏移

    float c = tex2D(texConstSrc,x,y);
    if (c != 0)
        iptr[offset] = c;//把热源温度复制到图像中(替换成原来的热源温度)
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *output_bitmap;
    float           *dev_inSrc;
    float           *dev_outSrc;
    float           *dev_constSrc;
    IMAGE           *bitmap;

    cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;
};


// clean up memory allocated on the GPU
void cleanup( DataBlock *d ) 
{
    cudaUnbindTexture( texIn );
    cudaUnbindTexture( texOut );
    cudaUnbindTexture( texConstSrc );
    HANDLE_ERROR( cudaFree( d->dev_inSrc ) );
    HANDLE_ERROR( cudaFree( d->dev_outSrc ) );
    HANDLE_ERROR( cudaFree( d->dev_constSrc ) );

    HANDLE_ERROR( cudaEventDestroy( d->start ) );
    HANDLE_ERROR( cudaEventDestroy( d->stop ) );
}


int main( void ) {
    DataBlock   data;
    IMAGE bitmap_image( DIM, DIM );
    data.bitmap = &bitmap_image;
    data.totalTime = 0;
    data.frames = 0;
    HANDLE_ERROR( cudaEventCreate( &data.start ) );
    HANDLE_ERROR( cudaEventCreate( &data.stop ) );

    int imageSize = bitmap_image.image_size();

    HANDLE_ERROR( cudaMalloc( (void**)&data.output_bitmap,
                               imageSize ) );

    // assume float == 4 chars in size (ie rgba)
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc,
                              imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc,
                              imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_constSrc,
                              imageSize ) );
    //通道格式描述符
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    HANDLE_ERROR( cudaBindTexture2D( NULL, texConstSrc,
                                   data.dev_constSrc,
                                   desc, DIM, DIM,
                                   sizeof(float) * DIM ) );

    HANDLE_ERROR( cudaBindTexture2D( NULL, texIn,
                                   data.dev_inSrc,
                                   desc, DIM, DIM,
                                   sizeof(float) * DIM ) );

    HANDLE_ERROR( cudaBindTexture2D( NULL, texOut,
                                   data.dev_outSrc,
                                   desc, DIM, DIM,
                                   sizeof(float) * DIM ) );

    // initialize the constant data
    float *temp = (float*)malloc( imageSize );
    for (int i=0; i<DIM*DIM; i++) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x>300) && (x<600) && (y>310) && (y<601))
            temp[i] = MAX_TEMP;
    }
    temp[DIM*100+100] = (MAX_TEMP + MIN_TEMP)/2;
    temp[DIM*700+100] = MIN_TEMP;
    temp[DIM*300+300] = MIN_TEMP;
    temp[DIM*200+700] = MIN_TEMP;
    for (int y=800; y<900; y++) {
        for (int x=400; x<500; x++) {
            temp[x+y*DIM] = MIN_TEMP;
        }
    }
    HANDLE_ERROR( cudaMemcpy( data.dev_constSrc, temp,
                              imageSize,
                              cudaMemcpyHostToDevice ) );    

    // initialize the input data
    for (int y=800; y<DIM; y++) {
        for (int x=0; x<200; x++) {
            temp[x+y*DIM] = MAX_TEMP;
        }
    }
    HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, temp,
                              imageSize,
                              cudaMemcpyHostToDevice ) );
    free( temp );

    int ticks=0;
    bitmap_image.show_image(30);
    while(1)
    {
        HANDLE_ERROR( cudaEventRecord( data.start, 0 ) );
        dim3    blocks(DIM/16,DIM/16);
        dim3    threads(16,16);
        IMAGE  *bitmap = data.bitmap;

        // since tex is global and bound, we have to use a flag to
        // select which is in/out per iteration
        volatile bool dstOut = true;
        for (int i=0; i<90; i++) {
            float   *in, *out;
            if (dstOut) {
                in  = data.dev_inSrc;
                out = data.dev_outSrc;
            } else {
                out = data.dev_inSrc;
                in  = data.dev_outSrc;
            }
            copy_const_kernel<<<blocks,threads>>>( in );
            blend_kernel<<<blocks,threads>>>( out, dstOut );
            dstOut = !dstOut;
        }
        float_to_color<<<blocks,threads>>>( data.output_bitmap,
                                            data.dev_inSrc );

        HANDLE_ERROR( cudaMemcpy( bitmap->get_ptr(),
                                data.output_bitmap,
                                bitmap->image_size(),
                                cudaMemcpyDeviceToHost ) );

        HANDLE_ERROR( cudaEventRecord( data.stop, 0 ) );
        HANDLE_ERROR( cudaEventSynchronize( data.stop ) );
        float   elapsedTime;
        HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                            data.start, data.stop ) );
        data.totalTime += elapsedTime;
        ++data.frames;
        printf( "Average Time per frame:  %3.1f ms\n",
                data.totalTime/data.frames  );

        ticks++;
        char key = bitmap_image.show_image(30);
        if(key==27)
        {
            break;
        }
    }

    cleanup(&data);
    return 0;
}

```
### 4.代码解析（下面是使用一维纹理内存的解析）
* 1.**申请纹理内存**：使用了浮点类型纹理内存的**引用**；**纹理内存必须声明为文件作用域内的全局变量！**

  ```C++
  //这些变量位于GPU上
  texture<float>  texConstSrc;
  texture<float>  texIn;
  texture<float>  texOut;
  ```


* **2.申请GPU全局内存**：下面代码为这三个缓存区分配了**GPU内存（全局内存）**,data.dev_inSrc等三个指针已经在结构对象data中声明了。

```C++
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc,
                              imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc,
                              imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_constSrc,
                              imageSize ) );
```

* **3.纹理内存与GPU全局内存绑定**：需要通过`cudaBindTexture()`将这些变量（上面的**纹理内存引用**）绑定到**内存缓冲区**。相当于告诉CUDA运行时两件事情：

  a. **指定的缓冲区**作为**纹理**来使用

  b.**纹理引用**作为纹理的"名字"

```C++
    HANDLE_ERROR( cudaBindTexture( NULL, texConstSrc,
                                   data.dev_constSrc,
                                   imageSize ) );
    HANDLE_ERROR( cudaBindTexture( NULL, texIn,
                                   data.dev_inSrc,
                                   imageSize ) );
    HANDLE_ERROR( cudaBindTexture( NULL, texOut,
                                   data.dev_outSrc,
                                   imageSize ) );
```

* **4.使用内置函数tex1Dfetch()**：当读取**核函数中**的纹理时，需要通过特殊的**函数**来告诉GPU**将读取请求转发到纹理内存而不是标准的全局内存**。`tex1Dfetch()`它是一个**编译器内置函数（Instrinsic）**。
* **5.使用二维纹理内存**：性能与一维的基本相同，但代码更简单。在使用内置函数**tex2Dfetch()**，读取缓存区中的数据时，不用计算缓存区中的线性偏置，而是可以直接用计算的像素索引x，y，**这样使得代码更为简洁，并且能自动处理边界问题**。
* **6.通道格式描述符**：在绑定二维纹理内存时，CUDA运行时要求提供一个**`cudaChanelFormatDesc()`**。在二维纹理内存的代码包含了一个对通道格式描述符的声明(Channel Format Descriptor)。在这里可以使用默认的参数，并且只要指定需要的是一个浮点描述符。然后我们通过1.**`cudaBindTexture2D()`**,2.纹理内存的位数（DIMXDIM）以及3.通道格式描述（desc）将这三个输入缓冲区绑定为二维纹理，main()函数其他部分保持不变。

## 纹理采样器（Texture Sampler），找不到该部分的内容？

如果使用了纹理采样器自动执行某种转换，那么纹理内存还能带来额外的加速。



# 第八章 互操作性

GPU的成功要归功于它能实时计算复杂的渲染任务，同时系统的其他部分还可以执行其他工作。

## 互操作性

### 概念：

* 通用计算：譬如前面的计算，在GPU上面进行的计算
* 渲染任务
* 互操作是指在通用计算与渲染模式之间互操作

### 提出问题：

问1：那么能否在**同一个应用程序中GPU既执行渲染计算，又执行通用计算？**

问2：如果要**渲染的图像**依赖于**通用计算**的结果，那么该如何处理？

问3：如果想要在**已经渲染的帧上**执行某种图像处理或者统计，又该如何实现？

## 与OpenGL的互操作性

CUDA程序生成图像数据传递给OpenGL驱动程序并进行渲染

### 1.代码：

```C++
/********************************************************************
*  SharedBuffer.cu
*  interact between CUDA and OpenGL
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
//下面两个头文件如果放反了会出错
#include "GL\glut.h"
#include "GL\glext.h"

#include <cuda_runtime.h>
//#include <cutil_inline.h>
#include <pcl\cuda\cutil_inline.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#define GET_PROC_ADDRESS(str) wglGetProcAddress(str)
#define DIM 512

PFNGLBINDBUFFERARBPROC    glBindBuffer = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData = NULL;

// step one:
// Step1: 申明两个全局变量，保存指向同一个缓冲区的不同句柄，指向要在OpenGL和CUDA之间共享的数据；
GLuint bufferObj;
cudaGraphicsResource *resource;


__global__ void cudaGLKernel(uchar4 *ptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	//将图像中心设为原点后的像素索引
	float fx = x / (float)DIM - 0.5f;
	float fy = y / (float)DIM - 0.5f;

	unsigned char green = 128 + 127 * sin(abs(fx * 100) - abs(fy * 100));

	ptr[offset].x = 0;
	ptr[offset].y = green;
	ptr[offset].z = 0;
	ptr[offset].w = 255;

}

void drawFunc(void)
{
	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

static void keyFunc(unsigned char key, int x, int y)
{
	switch (key){
	case 27:
		cutilSafeCall(cudaGraphicsUnregisterResource(resource));
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &bufferObj);
		exit(0);
	}
}

int main(int argc, char* argv[])
{
	// step 2:
	// 初始化CUDA
	// Step2: 选择运行应用程序的CUDA设备(cudaChooseDevice),告诉cuda运行时使用哪个设备来执行CUDA和OpenGL (cudaGLSetGLDevice）；
	cudaDeviceProp prop;
	int dev;

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cutilSafeCall(cudaChooseDevice(&dev, &prop));
	//为CUDA运行时使用openGL驱动做准备
	cutilSafeCall(cudaGLSetGLDevice(dev));

	//初始化OpenGL
	//在执行其他的GL调用之前，需要首先执行这些GLUT调用。
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("CUDA interact with OpenGL");


	glBindBuffer = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
	glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
	glGenBuffers = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
	glBufferData = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

	// Step3：在OpenGL中创建像素缓冲区对象；
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM*DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);//glBufferData()的调用需要OpenGL驱动程序分配一个足够大的缓冲区来保存DIM*DIM 个32位的值

	// step 4:
	// Step4: 通知CUDA运行时将像素缓冲区对象bufferObj注册为图形资源，实现缓冲区共享。
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));//cudaGraphicsMapFlagsNone表示不需要为缓冲区指定特定的行为
	                                                                                            //cudaGraphicsMapFlagsReadOnly将缓冲区指定为只读的
	                                                                                            //通过标志cudaGraphicsMapFlagsWriteDiscard来制定缓冲区之前的内容应该抛弃，从而使缓冲区变成只写的

	uchar4* devPtr;
	size_t size;
	cutilSafeCall(cudaGraphicsMapResources(1, &resource, NULL));
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	cudaGLKernel << <grids, threads >> >(devPtr);

	cutilSafeCall(cudaGraphicsUnmapResources(1, &resource, NULL));
	glutKeyboardFunc(keyFunc);
	glutDisplayFunc(drawFunc);
	glutMainLoop();
	return 0;
}


```

### 2.代码解析：

* Step1: 申明两个全局变量，保存指向同一个缓冲区的不同句柄，指向要在OpenGL和CUDA之间共享的数据；

```C++
	GLuint bufferObj;
	cudaGraphicsResource *resource;
```
* Step2: 选择运行应用程序的CUDA设备(cudaChooseDevice),告诉cuda运行时使用哪个设备来执行CUDA和OpenGL (cudaGLSetGLDevice）cutilSafeCall(cudaChooseDevice(&dev, &prop));


* Step3：**共享数据缓冲区**是在CUDA C核函数和OpenG渲染操作之间实现互操作的关键部分。要在OpenGL和CUDA之间传递数据，我们首先要创建一个缓冲区在这两组API之间使用，在OpenGL中创建像素缓冲区对象；，并将句柄保存在全局变量`GLuint bufferObj`中：

```C++
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM*DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);
```


* Step4: 通知CUDA运行时将像素缓冲区对象bufferObj注册为图形资源，实现缓冲区共享。
```C++
  cutilSafeCall(cudaGraphicsGLRegisterBuffer(&resource, 
                                             bufferObj,
                                             cudaGraphicsMapFlagsNone));
```
* 互操作性基本上就是调用接口，可以通过GPU Computing SDK的代码示例来学习

## 与DirectX的互操作性（略）



# 第九章 原子性

## 1.计算功能集

* 不同架构的CPU有着不同的功能和指令集（例如MMX、SSE(70条指令)、SSE2(使用了144个新增指令)等）
* 对于支持CUDA的不同图形处理器来说同样如此。NVIDIA将GPU支持的各种功能统称为**计算功能集（Compute Capability）**。

#### 基于最小功能集的编译

* 要支持全局内存原子操作，**计算功能集的最低版本为1.1**
* 当编译代码时，你需要告诉编译器：如果硬件支持的计算功能集版本低于1.1，那么将无法运行这个核函数。
* 要将这个信息告诉编译器，只需在调用NVCC时增加一个命令行选项：`nvcc -arch=sm_11`

* 当设置的计算能力比硬件本身高比如计算能力是6.1的（1080TI），设置 compute=62，sm=62 会出现错误，kernel不会被执行。
* 在.cu文件设置自己硬件的计算能力，如果不去设置或者去设置比较低的计算能力，比如设置compute_30,sm_30，那么自然地编译出来的程序的性能就会打折扣。

![1558098293977](C:\Users\xiaoxiong\AppData\Roaming\Typora\typora-user-images\1558098293977.png)

## 2.原子操作

示例：

x++；包含三步操作：a.读取x中的值；b.将步骤1中读到的值增加1；c.将递增后的结果写回到x。

现在考虑线程A和B都需要执行上面三个操作，**如果线程调度方式不正确，那么最终将（可能，因为六个步骤也可能会排出正确的结果）得到错误的结果**；

解决：

* 我们需要通过某种方式**一次性执行完读取-修改-写入这三个操作**，并且在执行过程中不会被其他线程所中断。**我们将满足这些条件限制的操作称为原子操作**。
* CUDA C支持多种原子操作，当有数千个线程在内存访问上发生竞争时，这些操作能够确保在内存上实现安全的操作。

## 3.计算直方图

概念：给定一个包含一组元素的数据集，直方图表示每个元素出现的频率。

在利用cpu实现的程序中，统计函数是：

```C++
	//统计
    for (int i=0; i<SIZE; i++)
        histo[buffer[i]]++;
```
**在GPU计算中，计算输入数组的直方图存在一个问题，即多个线程同时对输出直方图的同一个元素进行递增。在这种情况下，我们需要通过原子的递增操作来避免上面提到的问题**。

### 1.GPU代码：

```C++
//声明变量
 	unsigned int *dev_histo;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_histo,
                              256 * sizeof( int ) ) );
    //代码块内的变量一定要手动初始化
	HANDLE_ERROR( cudaMemset( dev_histo, 0,
                              256 * sizeof( int ) ) );
...............
__global__ void histo_kernel( unsigned char *buffer,
                              long size,
                              unsigned int *histo ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < size) {
        atomicAdd( &histo[buffer[i]], 1 );
        i += stride;
    }
}
..............
    
    // kernel launch - 2x the number of mps gave best timing
    cudaDeviceProp  prop;
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ) );
    int blocks = prop.multiProcessorCount;
    histo_kernel<<<blocks*2,256>>>( dev_buffer, 
                                    SIZE, 
                                   dev_histo );
```

* 这里的atomicAdd就是同时只能有一个线程操作，防止了其他线程的骚操作。
* 引入了一个新的CUDA运行时函数，**`cudaMemset()`**函数，用于**内存空间初始化**。
* 由于直方图包含了256个元素，因此可以在每个线程块中包含256个线程
* 通过一些性能实验，我们发现当线程块的数量为GPU数量的2倍是，将达到最佳性能。
* 由于核函数中只包含非常少的计算工作，因此很可能是全局内存上的原子操作导致性能的降低，**当数千个线程尝试访问少量的内存位置是，将发生大量的竞争**。

### 2.改进版：

* 使用**共享内存**和**全局内存原子操作**的直方图核函数

```C++
#define SIZE    (100*1024*1024)


__global__ void histo_kernel( unsigned char *buffer,
                              long size,
                              unsigned int *histo ) {

    __shared__  unsigned int temp[256];//声明一个共享缓冲区
    temp[threadIdx.x] = 0;             //将清除内存，每个线程写一次，由于我们在核函数设置启动线程中
    								   //为每个block分配了256个线程，所以很容易清除累计缓冲区temp。
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;//因为线程数没有数据多所以要设定步长（步长为分配的线程数目）
    while (i < size) {
        atomicAdd( &temp[buffer[i]], 1 );
        i += stride;
    }

    __syncthreads();
    atomicAdd( &(histo[threadIdx.x]), temp[threadIdx.x] );
}

·············
    cudaDeviceProp  prop;
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ) );
    int blocks = prop.multiProcessorCount;
    histo_kernel<<<blocks*2,256>>>( dev_buffer,
                                    SIZE, 
                                   dev_histo );
```

* 在共享内存中计算这些直方图，这将避免每次将写入操作从芯片发送到DRAM，现在只有256个线程在256个地址上发生竞争，这将极大地减少在全局内存中数千个线程之间发生竞争的情况。



# 第十章 流

**通过CUDA流在GPU上用任务并行**

## 页锁定主机内存

两个**主机内存**分配函数：

1. **标准C库函数`malloc()`**在主机上分配内存
2. **CUDA运行时**提供自己独有的机制来分配**主机内存**：**`cudaHostAlloc()`**。

两个函数分配的**内存之间的差异**：

1. **`malloc()`**将分配标准的，可分页的（Pagable）主机内存，
2. **`cudaHostAlloc()`**将分配页锁定的主机内存（固定内存）

### 页锁定主机内存

页锁定主机内存也称为**固定内存（Pinned Memory）**或者不可分内存。

1. 对于固定内存，**操作系统将不会对这块内存分页交换到磁盘上，从而确保了该内存始终驻留在物理内存中。因此，操作系统能够安全地使用某个程序访问该内存的物理地址，因为这块内存将不会被破坏或者重新定位。**--->物理地址固定不变。
2. 由于知道内存的物理地址，因此可以通过“**直接内存访问**(Direct Memory Access,DMA)”技术来在GPU和主机之间复制数据。DMA操作在可分页内存中可能会延迟--->DMA复制过程中使用固定内存非常重要，**页锁定主机内存（固定内存）的性能比标准可分页的性能要高大约2倍。**

实际上并不是说使用固定内存就好



![1558171366351](C:\Users\xiaoxiong\AppData\Roaming\Typora\typora-user-images\1558171366351.png)

3. 固定内存是一把双刃剑。**但是用固定内存时，你将失去虚拟内存的所有功能。**应用程序中使用每个固定内存时都需要**分配物理内存**，因为这些内存不能交换到磁盘上。--->意味着系统更快地耗尽内存。

4. 使用情况：仅对`cudaMemcpy()`调用中的源内存或者目标内存，才使用也锁存内存，并且不再需要他们时立即释放，而不是等到程序关闭才释放。

5. 页锁定内存的作用不仅限于性能的提升，后面章节会看到，在一些特殊情况中也需要使用页锁定内存。

6. 调用：

   ```C++
   	
   	#define SIZE    (64*1024*1024)
   	int             *a；
   	int size = SIZE;
   	//CUDA运行时申请固定内存
       HANDLE_ERROR( cudaHostAlloc( (void**)&a,
                                    size * sizeof( *a ),
                                    cudaHostAllocDefault ) );
   ```

   

## 计算带宽

```C++
float           MB = (float)100*SIZE*sizeof(int)/1024/1024;//SIZE=(64*1024*1024)
printf( "\tMB/s during copy up:  %3.1f\n",
            MB/(elapsedTime/1000) );//elapsedTime=用时
```



## CUDA流

* **CUDA流表示一个操作GPU队列**
* 该队列的操作将以指定的顺序执行。我们可以在流中添加一些操作，例如启动核函数，内存复制，以及事件的启动和结束等。
* **可以将流视为GPU上的一个任务，并且这些任务可以并行执行**。

### 设备重叠功能的GPU

支持设别重叠功能的GPU能在**执行一个CUDA C核函数**的同时，还能在**设备与主机之间执行复制操作**。可以使用多个流来实现这种计算与数据传输的重叠。

### 使用流

```C++
#include "../common/book.h"

#define N   (1024*1024)
#define FULL_DATA_SIZE   (N*20)

//核函数N个线程，每次处理N个数
__global__ void kernel( int *a, int *b, int *c ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}
        

int main( void ) {
    cudaDeviceProp  prop;
    int whichDevice;
    HANDLE_ERROR( cudaGetDevice( &whichDevice ) );
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, whichDevice ) );
    //判断：支持设别重叠功能
    if (!prop.deviceOverlap) {
        printf( "Device will not handle overlaps, so no speed up from streams\n" );
        return 0;
    }
	//创建事件和流
    cudaEvent_t     start, stop;
    float           elapsedTime;

    cudaStream_t    stream;
    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c;

    // start the timers
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );

    // initialize the stream
    HANDLE_ERROR( cudaStreamCreate( &stream ) );


    // 分配设备内存，只申请了20分之一的数据量大小的内存
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a,
                              N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b,
                              N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c,
                              N * sizeof(int) ) );


    // 在这里申请固定内存不仅仅是为了让复制操作执行得更快
    // 要以异步的方式在主机和设备之间复制数据必须是固定内存
    // 申请内存大小为数据大小
    HANDLE_ERROR( cudaHostAlloc( (void**)&host_a,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&host_b,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&host_c,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );
    // 填充申请的缓冲区host_a，host_b
    for (int i=0; i<FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    // now loop over full data, in bite-sized chunks

	//我们不将输入缓冲区整体复制到GPU，而是将输入缓冲区划分成更小的块（分成20块），并在每个块上执行一个包含三个步骤的过程：
    //1.将一部分输入缓冲区复制到GPU ；2.在这部分缓冲区上运行核函数；3.然后将一部分输入缓冲区复制到GPU
	
    for (int i=0; i<FULL_DATA_SIZE; i+= N) {
        // copy the locked memory to the device, async
        // 将固定内存以异步的方式复制到设备上
        HANDLE_ERROR( cudaMemcpyAsync( dev_a, host_a+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream ) );
        HANDLE_ERROR( cudaMemcpyAsync( dev_b, host_b+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream ) );

        // 核函数带有流参数
        // 刚好N个线程N个数据，线程不需要多次工作
        kernel<<<N/256,256,0,stream>>>( dev_a, dev_b, dev_c );

        // 将数据从设备复制到锁定内存
        HANDLE_ERROR( cudaMemcpyAsync( host_c+i, dev_c,
                                       N * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream ) );

    }
    // copy result chunk from locked to full buffer
    HANDLE_ERROR( cudaStreamSynchronize( stream ) );

    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );

    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time taken:  %3.1f ms\n", elapsedTime );

    // cleanup the streams and memory
    HANDLE_ERROR( cudaFreeHost( host_a ) );
    HANDLE_ERROR( cudaFreeHost( host_b ) );
    HANDLE_ERROR( cudaFreeHost( host_c ) );
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_c ) );
    HANDLE_ERROR( cudaStreamDestroy( stream ) );
	getchar();
    return 0;
}
//Time taken:  25.4 ms
```

* 创建流和事件
* 分配好设备内存和主机内存
* 分块执行三个步骤
* 当for循环结束时，队列中应该包含了很多等待GPU执行的工作。如果想要确保GPU只能执行完了计算和内存复制等操作。那么就需要将**GPU与主机同步**。也就是说主机在继续执行之前要先等待GPU完成。调用**`cudaStreamSynchronize()`**并指定想要等待的流

### 主机与设备之间复制数据

1. `cudaMemcpy()`同步方式执行：意味着，当函数返回时，复制操作已经完成，并且在输出缓冲区包含了复制进去的内容。
2. 新函数`cudaMemcpyAsync()`异步方式执行：与同步方式相反，在调用该函数时，只是放置一个请求，表示在流中执行一次内存复制操作，这个流是通过函数stream来指定的。**当函数返回时，我们无法确保复制操作是否已经启动，更无法保证它是否已经结束。我们能够保证的是，复制操作肯定会当下一个被放入流中的操作之前执行**。
3. 任何一个传递给`cudaMemcpyAsync()`的主机内存指针都必须已经通过`cudaHostAlloc()`分配好内存。**你只能已异步方式对固定内存进行复制操作。**

### 带有流参数的核函数

此时核函数的调用**是异步的**。

## 使用多个流

![cuda流](https://img-blog.csdn.net/20170716103917013?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvRmlzaFNlZWtlcg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

改进思想：

1. 分块计算
2. 内存复制和核函数执行的重叠

* 上图中，第0个流执行：核函数时，在第1个流中执行：输入缓冲区复制到GPU......
* 在任何支持内存复制和核函数的执行相互重叠的设备上，当使用多个流是，应用程序的整体性能都会提升。

代码：

```C++
#include "../common/book.h"

#define N   (1024*1024)
#define FULL_DATA_SIZE   (N*20)


__global__ void kernel( int *a, int *b, int *c ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}


int main( void ) {
    cudaDeviceProp  prop;
    int whichDevice;
    HANDLE_ERROR( cudaGetDevice( &whichDevice ) );
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, whichDevice ) );
    if (!prop.deviceOverlap) {
        printf( "Device will not handle overlaps, so no speed up from streams\n" );
        return 0;
    }
	//事件
    cudaEvent_t     start, stop;
    float           elapsedTime;
	//流
	cudaStream_t    stream0, stream1;

    int *host_a, *host_b, *host_c;
    int *dev_a0, *dev_b0, *dev_c0;
    int *dev_a1, *dev_b1, *dev_c1;


    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaStreamCreate( &stream0 ) );
    HANDLE_ERROR( cudaStreamCreate( &stream1 ) );

    // 申请内存
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a0,
                              N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b0,
                              N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c0,
                              N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a1,
                              N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b1,
                              N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c1,
                              N * sizeof(int) ) );
	
    //申请页锁定内存
    HANDLE_ERROR( cudaHostAlloc( (void**)&host_a,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&host_b,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&host_c,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );

	//初始化页锁定内存
    for (int i=0; i<FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    // now loop over full data, in bite-sized chunks
    for (int i=0; i<FULL_DATA_SIZE; i+= N*2) {
        // stream0
        HANDLE_ERROR( cudaMemcpyAsync( dev_a0, host_a+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream0 ) );
        HANDLE_ERROR( cudaMemcpyAsync( dev_b0, host_b+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream0 ) );

        kernel<<<N/256,256,0,stream0>>>( dev_a0, dev_b0, dev_c0 );

        HANDLE_ERROR( cudaMemcpyAsync( host_c+i, dev_c0,
                                       N * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream0 ) );


        // stream1
        HANDLE_ERROR( cudaMemcpyAsync( dev_a1, host_a+i+N,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream1 ) );
        HANDLE_ERROR( cudaMemcpyAsync( dev_b1, host_b+i+N,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream1 ) );

        kernel<<<N/256,256,0,stream1>>>( dev_a1, dev_b1, dev_c1 );


        HANDLE_ERROR( cudaMemcpyAsync( host_c+i+N, dev_c1,
                                       N * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream1 ) );

    }
    //两个流都要将CPU与GPI同步。
    HANDLE_ERROR( cudaStreamSynchronize( stream0 ) );
    HANDLE_ERROR( cudaStreamSynchronize( stream1 ) );
	//
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );

    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time taken:  %3.1f ms\n", elapsedTime );

    // cleanup the streams and memory
    HANDLE_ERROR( cudaFreeHost( host_a ) );
    HANDLE_ERROR( cudaFreeHost( host_b ) );
    HANDLE_ERROR( cudaFreeHost( host_c ) );
    HANDLE_ERROR( cudaFree( dev_a0 ) );
    HANDLE_ERROR( cudaFree( dev_b0 ) );
    HANDLE_ERROR( cudaFree( dev_c0 ) );
    HANDLE_ERROR( cudaFree( dev_a1 ) );
    HANDLE_ERROR( cudaFree( dev_b1 ) );
    HANDLE_ERROR( cudaFree( dev_c1 ) );


    HANDLE_ERROR( cudaStreamDestroy( stream0 ) );
    HANDLE_ERROR( cudaStreamDestroy( stream1 ) );


	getchar();
    return 0;
}
```

* 因为使用了两个流，for循环中处理的数据量为原来的两倍，步长为原来的两倍，程序处理的总数据量不变。

* 处理数据量是相同的，结果是一个流与两个流使用的时间差不多。

  一个流使用的时间是24.1ms～25.2ms，

  两个流使用的时间是：23.1～23.9ms，

  修改后代码使用时间为23.9ms～24.9ms

* 使用了流的确改善了执行时间，但是在一个流和多个流之间并没有明显的性能提高。

## GPU工作调度机制

* 程序员可以将流视为有序的操作序列，其中既包含内存复制操作，又包含核函数调用。
* 然而，硬件中并没有流的概念，而是**包含一个或多个引擎来执行内存复制操作**，以及**一个引擎来执行核函数**。这些引擎彼此独立地对操作进行排队。

![技术分享](http://image.bubuko.com/info/201704/20180110230842521801.png)

应用程序首先将第0个流的所有操作放入队列，然后是第一个流的所有操作。CUDA驱动程序负责按照这些操作的顺序把他们调度到硬件上执行，这就维持了流内部的依赖性。图10.3说明了这些依赖性，**箭头表示复制操作要等核函数执行完成之后才能开始**。

![技术分享](http://image.bubuko.com/info/201704/20180110230842524731.png)

于是得到这些操作在硬件上执行的时间线：

![ææ¯åäº"](http://image.bubuko.com/info/201704/20180110230842525708.png)



* 图中显示，**第0个流复制C**阻塞了**第1个流复制A,第一个流复制B**，导致**第0个流执行完核函数**还要等待内存复制引擎完成流0复制C，流1复制A，流1复制B的三个操作才能执行**流1核函数**

由于第0个流中将c复制回主机的操作要等待核函数执行完成，因此第1个流中将a和b复制到GPU的操作虽然是完全独立的，但却被**阻塞了，这是因为GPU引擎是按照指定的顺序来执行工作**。记住，硬件在处理内存复制和核函数执行时分别采用了不同的引擎，因此我们需要知道，将操作放入流队列中的顺序将影响着CUDA驱动程序调度这些操作以及执行的方式。

## 高效使用多个流

如果同时调度某个流的所有操作，那么很容易在无意中阻塞另一个流的复制操作或者核函数执行。要解决这个问题，在将操作放入流的队列时应采用宽度优先方式，而非深度优先方式。如下代码所示：

```C++
  for (int i=0; i<FULL_DATA_SIZE; i+= N*2) {
        // enqueue copies of a in stream0 and stream1
        HANDLE_ERROR( cudaMemcpyAsync( dev_a0, host_a+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream0 ) );
        HANDLE_ERROR( cudaMemcpyAsync( dev_a1, host_a+i+N,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream1 ) );
        // enqueue copies of b in stream0 and stream1
        HANDLE_ERROR( cudaMemcpyAsync( dev_b0, host_b+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream0 ) );
        HANDLE_ERROR( cudaMemcpyAsync( dev_b1, host_b+i+N,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream1 ) );

        // enqueue kernels in stream0 and stream1   
        kernel<<<N/256,256,0,stream0>>>( dev_a0, dev_b0, dev_c0 );
        kernel<<<N/256,256,0,stream1>>>( dev_a1, dev_b1, dev_c1 );

        // enqueue copies of c from device to locked memory
        HANDLE_ERROR( cudaMemcpyAsync( host_c+i, dev_c0,
                                       N * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream0 ) );
        HANDLE_ERROR( cudaMemcpyAsync( host_c+i+N, dev_c1,
                                       N * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream1 ) );
    }
```

如果内存复制操作的时间与核函数执行的时间大致相当，那么新的执行时间线将如图10.5所示，在新的调度顺序中，依赖性仍然能得到满足：

![技术分享](http://image.bubuko.com/info/201704/20180110230842528637.png)

由于采用了宽度优先方式将操作放入各个流的队列中，因此第0个流对c的复制操作将不会阻塞第1个流对a和b的内存复制操作。这使得GPU能够并行的执行复制操作和核函数，从而使应用程序的运行速度显著加快。

* 实验结果表明，并没有改进性能，可能是**高版本的CUDA运行时已经对流和复制引擎等进行了优化（个人猜想）**

# 第十一章 多GPU

## 零拷贝主机内存

* 前面使用函数`cudaHostAlloc()`申请固定内存，并且设定参数`cudaHostAllocDefault`来获得默认的固定内存。
* 在函数`cudaHostAlloc()`使用其他参数值：**`cudaHostAllocMapped`**分配的主机内存也是固定的，它与通过`cudaHostAllocDefault`分配的固定内存有着相同的属性，特别当它不能从物理内存中交换出去或者重新定位时。
* 这种内存除了可以用于主机和GPU之间的内存复制外，还可以打破第三章主机内存规则之一：**可以在CUDA C核函数中直接访问这种类型的主机内存。由于这种内存不需要复制到GPU，因此也称为零拷贝内存**

```c++
float cuda_pinned_alloc_test(int size) {
	cudaEvent_t start, stop;
	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate the memory on the CPU
	cudaHostAlloc((void**)&a, size * sizeof(float),
		cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&b, size * sizeof(float),
		cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&partial_c, blocksPerGrid * sizeof(float),
		cudaHostAllocMapped);

	// find out the GPU pointers
	cudaHostGetDevicePointer(&dev_a, a, 0);
	cudaHostGetDevicePointer(&dev_b, b, 0);
	cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0);

	// fill in the host memory with data
	for (int i = 0; i < size; i++) {
		a[i] = i;
		b[i] = i * 2;
	}

	cudaEventRecord(start, 0);

	dot << <blocksPerGrid, threadsPerBlock >> >(size, dev_a, dev_b, dev_partial_c);

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	// finish up on the CPU side
	c = 0;
	for (int i = 0; i < blocksPerGrid; i++) {
		c += partial_c[i];
	}
	//无论使用什么标志都使用这个函数来释放
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(partial_c);

	// free events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("计算结果:  %f\n", c);

	return elapsedTime;
}
```

*  **`cudaHostAllocMapped`**：这个标志告诉运行时将从GPU中访问这块内存（分配零拷贝内存）
* **`cudaHostAllocWriteCombined`**：这个表示，运行时将内存分配为“合并式写入（Write-Combined）”内存。这个标志并不会改变应用程序的功能，但却可以显著地提升GPU读取内存时的性能。然而CPU也要读取这块内存时，“合并式写入”会显得很低效。
* **`cudaHostGetDevicePointer`**：获取这块内存在GPU上的有效指针。这些指针将被传递给核函数。
* **`cudaThreadSynchronize()`**：将CPU与GPU同步，在同步完成后面就可以确信核函数已经完成，并且在零拷贝内存中包含了计算好的结果。

### 零拷贝内存的性能

* 所有固定内存都存在一定的局限性，零拷贝内存同样不例外：**每个固定内存都会占用系统的可用物理内存，这终将降低系统的性能。**(只用在使用一次的情况的原因)
* 使用零拷贝内存通常会带来性能的提升，因为内存在物理上与主机是共存的。将缓冲区声明为零拷贝内存的唯一作用就是避免不必要的数据复制。
* 当输入内存和输出内存都只是用一次时，那么独立GPU上使用零拷贝内存将带来性能提升。如果多次读取内存，那么最终将得不偿失，还不如一开始将数据复制到GPU。

## 使用多个GPU

NVIDIA一个显卡可能包含多个GPU。例如GeForce GTX 295、Tesla K10。虽然GeForce GTX 295物理上占用一个扩展槽，但在CUDA应用程序看来则是两个独立的GPU。

将多个GPU添加到独立的PCIE槽上，通过NVIDIA的SLI技术将他们桥接。

略。。。

## 可移动的固定内存---使得多个GPU共享固定内存

问题：

固定内存实际上是主机内存，只是该内存页锁定在物理内存中，以防止被换出或重定位。然而这些内存页**仅对于单个GPU线程（书上写的是单个CPU线程）来说是“固定的”**，如果某个线程分配了固定内存，那么这些内存页只是对于分配它们的线程来说是页锁定的。**如果在线程之间共享指向这块内存的指针，那么其他的线程把这块内存视为标准的、可分页的内存**。

副作用：当其他线程（不是分配固定内存的线程）试图在这块内存上执行`cudaMemcpy()`时，将按照标准的可分页内存速率来执行复制操作。这种速率大约为最高传输速度的50%。更糟糕的时，如果线程视图将一个`cudaMemcpyAsync()`调用放入CUDA流的队列中，那么将失败，因为`cudaMemcpyAsync()`需要使用固定内存。由于这块内存对于除了分配它线程外的其他线程来说视乎是可分页的，因为这个调用会失败，甚至导致任何后续操作都无法进行。

解决：

可以将固定内存分配为可移动，这意味着可以在主机线程之间移动这块内存，并且每个线程都将其视为固定内存。

* 使用`cudaHostAlloc()`来分配内存，并在调用时使用一个新标志：**`cudaHostAllocPortable`**这个标志可以与其他标志一起使用，例如`cudaHostAllocWriteCombined`和`cudaHostAllocMapped`这意味着分配主机内存时，可以将其作为可移动，零拷贝以及合并写入等的任意组合。



# 第十二章 后记

## CUDA工具

1. CUFFT：快速傅立叶变换
2. CUBLAS：线性代数函数
3. 实例程序：NVIDIA GPU Computing SDK

![1558370188960](C:\Users\xiaoxiong\AppData\Roaming\Typora\typora-user-images\1558370188960.png)

4. NVIDIA性能原语（NPP）：高性能图像处理或视频应用程序
5. 调试工具NVIDIA Parallel Nsight



# 其他



## 设备信息

Device: <GeForce GTX 1080 Ti> canMapHostMemory: Yes

CUDA Capable: SM 6.1 hardware

28 Multiprocessor(s) x 128 (Cores/Multiprocessor) = 3584 (Cores)




