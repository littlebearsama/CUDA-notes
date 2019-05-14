# 第三章 简介

- 将CPU即系统的内存称为主机（**host**），而将GPU及其内存称为设备（**device**）

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

| 限定符                     | 在哪里被调用                | 在哪里被执行 |
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

**`__constant__`** 限定符可选择与 **`__device__`**限定符一起使用，所声明的变量具有以下特征：1.位于固定存储器空间中；2. 与应用程序具有相同的生命周期；3.可通过网格内的所有线程访问，也可通过运行时库从主机访问。

**`__shared__`** 限定符可选择与 **`__device__`**限定符一起使用，所声明的变量具有以下特征：1.位于线程块的共享存储器空间中；2. 与块具有相同的生命周期；3.尽可通过块内的所有线程访问。只有在` _syncthreads()`_的执行写入之后，才能保证共享变量对其他线程可见。除非变量被声明为瞬时变量，否则只要之前的语句完成，编译器即可随意优化共享存储器的读写操作。

## 2.参数传递

- `<<<>>>`尖括号表示要将一些参数传递给运行时系统，**这些参数并不是传递给设备代码的参数**，而是告诉运行时**如何启动设备代码**。传递给设备代码本身的参数是放在圆括号中传递的。

  > 尖括号作用？**线程配置**。
  >
  > <<<Dg, Db, Ns, S>>>
  >
  > 1. Dg 的类型为 dim3，指定网格的维度和大小，**Dg.x * Dg.y 等于所启动的块数量**，Dg.z =1无用，目前还不支持三维的线程格；如果指定Dg=256，那么将有256个线程块在GPU上运行。
  > 2. Db 的类型为 dim3，指定各块的维度和大小，Db.x * Db.y * Db.z **等于各块的线程数量**；
  > 3. Ns 的类型为 size_t，指定各块为此调用动态分配的共享存储器（除静态分配的存储器之外），这些动态分配的存储器可供声明为外部数组的其他任何变量使用，Ns 是一个可选参数，默认值为 0；
  > 4. S 的类型为 cudaStream_t，指定相关流；S 是一个可选参数，默认值为 0。

- 核函数内部可以调用CUDA**内置变量**，比如threadIdx，blockDim等。下下章将具体谈到线程索引。
- 参数传递和普通函数一样，**通过括号内的形参传递。**

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

- 调用cudaMalloc()在**设备上**为三个数组分配内存。
- 使用完GPU后调用cudaFree()来释放他们。
- 通过cudaMemcpy()进行主机与设备之间复制数据。



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

- 简单来说就是启动了充足的线程，而有的线程不用工作，为了防止核函数不会出现越界读取等错误，我们使用了条件判断if（tid<N）。

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

- blocks和threads是两个二维变量
- 由于生成的是一幅图像，因此使用二维索引，并且每个线程都有唯一的索引`(x,y)`，这样可以很容易与输出图像中的像素一一对应起来。
- 加入线程块是一个16X16的线程数组，图像有DIMXDIM个像素，那么就需要启动DIM/16 x DIM/16个线程块，从而使每一个像素对应一个线程。
- GPU优势在于处理图像时比如1920X1080需要创建200万个线程，CPU无法完成这样的工作。

## 4.共享内存和同步

- 共享内存术语Shared Memory，是位于SM（流多处理器）中的特殊存储器。还记得SM吗，就是流多处理器，大核是也。
- 将关键字**`__share__`**添加到变量声明中，这将是这个变量**驻留**在**共享内存**中。
- block与block的线程无法通信
- 共享内存缓存区驻留在物理GPU上，而不是驻留在GPU以外的系统内存中。因此，**在访问共享内存时的延迟要远远低于访问普通缓存区的延迟**，使得共享内存像**每个线程块的高速缓存**或中间结果暂存器那样高效。
- **想要在线程之间通信，那么还需要一种机制来实现线程之间的同步**，例如，如果**线程A**将一个值写入到共享内存，并且我们希望**线程B**对这个值进行一些操作，那么只有当线程A写入操作完成后，线程B才能开始执行它的操作。**如果没有同步，那么将发生竞态条件**。

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

- 当某些线程需要执行一条指令，而其他线程不需要执行时，**这种情况就称为线程发散（Thread Divergence）**。在正常环境中，发散的分支只会使得某些线程处于空闲状态，而其他线程将执行分支中的代码。但是在**`__syncthreads()`**情况中，线程发散造成的结果有些糟糕。CUDA架将确保，除非线程块中的每个线程都执行了**`__syncthreads()`**，否则没有任何线程能执行**`__syncthreads()`**之后的指令。如果**`__syncthreads()`**位于发散分支中，一些线程将永远无法执行**`__syncthreads()`**。硬件将一直保持等待。
- 下面代码将使处理器挂起，因为GPU在等待某个永远无法发生的事件。

```C++
        if (cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
          __syncthreads();
          }
```



- 例子2（二维线程布置）基于共享内存的位图（略）

# 第六章 常量内存与事件

# 第七章 纹理内存

# 第八章 互操作性

# 第九章 原子性

# 第十章 流

# 第十一章 多GPU

# 第十二章 后记







# 第九章 原子性操作

原子性操作，就是，像操作系统的PV操作一样，同时只能有一个线程进行。好处自然是不会产生同时读写造成的错误，坏处显而易见是增加了程序运行的时间。

## 计算直方图

原理：假设我们要统计数据范围是[0,255]，因此我们定义一个`unsigned int histo[256]`数组，然后我们的数据是`data[N]`，我们遍历data数组，然后`histo[data[i]]++`，就可以在最后计算出直方图了。这里我们引入了原子操作

```C++
__global__ void histo_kernel(unsigned char *buffer, long size,
        unsigned int *histo) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < size) {
        atomicAdd(&(histo[buffer[i]]), 1);
        i += stride;
    }
}
```

这里的atomicAdd就是同时只能有一个线程操作，防止了其他线程的骚操作。但是，巨慢，书里说自从服用了这个，竟然比CPU慢四倍。因此我们需要别的。

## 升级版计算直方图

使用原子操作很慢的原因就在于，当数据量很大的时候，会同时有很多对于一个数据位的操作，这样操作就在排队，而这次，我们先规定线程块内部有256个线程(这个数字不一定)，然后在线程内部定义一个临时的共享内存存储临时的直方图，然后最后再将这些临时的直方图加总。这样冲突的范围从全局的所有的线程，变成了线程块内的256个线程，而且由于也就256个数据位，这样造成的数据冲突会大大减小。具体见以下代码：

```C++
__global__ void histo_kernel(unsigned char *buffer, long size,
        unsigned int *histo) {
    __shared__ unsigned int temp[256];
    temp[threadIdx.x] = 0;
    //这里等待所有线程都初始化完成
    __syncthreads();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    while (i < size) {
        atomicAdd(&temp[buffer[i]], 1);
        i += offset;
    }
    __syncthreads();
    //等待所有线程完成计算，讲临时的内容加总到总的直方图中
    atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}
```

# 第十章 流

1. 页锁定内存 
   这种内存就是在你申请之后，锁定到了主机内存里，它的物理地址就固定不变了。这样访问起来会让效率增加。
2. CUDA流 
   流的概念就如同java里多线程的概念一样，你可以把不同的工作放入不同的流当中，这样可以并发执行一些操作，比如在内存复制的时候执行kernel: 

![cuda流](https://img-blog.csdn.net/20170716103917013?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvRmlzaFNlZWtlcg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



文后讲了一些优化的方法，但是亲测无效啊，可能是cuda对于流的支持方式变了，关于流的知识会在以后的博文里再提及。







