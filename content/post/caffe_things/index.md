---
title: Caffe使用问题记录
date: 2018-08-07
categories:
    - 技术经验
    - 问题记录
tags:
    - caffe
---
以往在使用 caffe 中遇到的部分问题记录。
<!--more-->

## 使用教程
Caffe 一般通过编译生成的可执行文件 caffe（一般路径为 `$CAFFE_PATH/build/tools/caffe`）来进行网络训练和测试。

### TL;DR
1. Python 调用（pycaffe 路径 `caffe/python/caffe`）
```python
  import caffe
  net = caffe.Net(prototxt, caffemodel, TEST) # 设置网络
  net.blobs['对应层的name'].data[...] = input # 操作输入输出
  pred = net.foward() # 前向取出最终层结果, 也可以通过  pred = net.blobs['name'].data 来拿到
```
2. C++ 前向追踪：  
 `net.cpp::Forward -> layer.hpp::Forward -> 各layer的Forward (src/caffe/layers/*.cpp & *.cu)`


自定义修改：
1. 自定义网络结构 ( 根据 `caffe.proto` ):
    - 改 `train.prototxt`、`solver.prototxt`、`test.prototxt`
1. 自定义层:
    - 在 `src/caffe/layers/*` 下新增 .cpp、.cu
    - 在 `include/caffe/layers*` 下新增 .hpp
    - 改 `caffe.proto`，增加参数条目
    - 部分复杂的子函数实现会放在 `src/caffe/util` 内


### 训练所需文件
- `*.prototxt`：用于定义网络结构的文件，一般在网络本身的基础上加入了训练和测试过程所需的网络模块，以及模块相应的训练和测试用参数。
- `*_deploy.prototxt`：同样是定义网络结构的文件，但只包含了前向推理部分，没有训练部分的模块和参数。
- `*_solver.prototxt`：用于训练和测试的配置文件，类似学习率、学习策略、惩罚项和输出信息等，可在 `$CAFFE_PATH/src/caffe/proto/caffe.proto` 中的 `SolverParameter` 找到具体配置项信息。  

上述文件的名字可随意更改，不过一般会加上这些后缀用以区分。

### 路径相关
训练时需要指定 `*_solver.prototxt` 的路径，并在 `*_solver.prototxt` 指明网络 `*.prototxt` 的路径（也可以分开指定训练和测试用的网络），还需要定义输出模型以及状态的路径前缀 `snapshot_prefix`。  
一般 `*.prototxt` 里还需要在输入数据层指明输入数据集的路径。  

### 指令相关
训练：（可以加上预训练模型 `-weights /路径/*.caffemodel` 或 恢复训练状态 `-snapshot /路径/*.solverstate`，`*.solverstate` 是在 `*.caffemodel` 基础上加上了训练的状态信息）

	$CAFFE_PATH/build/tools/caffe train -solver *_solver.prototxt

测试：（需要指定训练测试用的网络和训练好的模型，测试的样本数为 TEST_PHASE 的 `batch_size` x `iterations`）

	$CAFFE_PATH/build/tools/caffe test -model *.prototxt -weights *.caffemodel -iterations 100 -gpu 0

前向：用 python 接口调用 caffe 完成，基本流程形如
```python
import caffe
net = caffe.Net(prototxt_path, weights_path, caffe.TEST)
net.blobs['data'].data[...] = input_image
net.forward()
label = net.blobs['label'].data
```

### Caffe 网络相关
Caffe 的网络 `Net` 是由各种层 `Layer` 组成的有向无环图，网络层通过 `Blob` 来存储 feature maps，具体层的实现在 `$CAFFE_PATH/src/caffe/layers/` 和 `$CAFFE_PATH/include/caffe/layers/` 里，其传递参数定义于 `$CAFFE_PATH/src/caffe/proto/caffe.proto`。网络结构文件中有什么看不懂或者需要深入了解的，找这三个地方就对了。  

训练网络一般包含训练过程所需要的所有部分，比如数据输入层（如 Data Layer），Loss 层（如 SoftmaxWithLoss Layer），定义好后 Caffe 会自己输入数据然后计算输出 loss 并更新参数。前向网络同理。

如果需要用到 Caffe 没有的自定义网络层，需要自己编写相应 C++/CUDA 代码，放置于上述两个文件夹中，如果有传入参数还需要在 `caffe.proto` 中添加相应需要传入的参数配置。对于不好实现而且不需要 gpu 加速的自定义层，可以通过 python layer 来实现。  

对于有复杂操作的网络，比如 loss 需要在外部计算，则可以通过 Caffe 的 python 接口实现，在 python 环境中做训练更新。  

### 关于分割网络的训练说明
数据输入层 `DenseImageData` 是自定义层，在 `dense_image_data_param` 的 `source` 中指明了输入图片集的路径，txt 文件内的格式是：  
"样本图路径 标签路径"  

标签与样本图同等大小（224x224），单通道，其中像素值 0 为前景，1 为背景。

loss_s8、loss 的 class_weighting 为对应标签的 loss 权重，用于解决样本不平衡。class_weighting 计算方法：对所有数据集标签统计各类别个数，比如 0 的个数，1 的个数。`class_weighting_0 = num(1)/(num(1)+num(0))`、`class_weighting_1 = num(0)/(num(1)+num(0))`。

## 缺陷记录
- `Xavier`初始化没有乘上增益 (ReLU应乘根号2, 等等)
- 在matlab上训练得出的模型是col-major,需要将所有矩阵参数转置才能在其他地方用
- 老版本caffe在初次前向时会比较慢, 新版未知
- caffe 初始化数据层时启动线程是 __TEST__ 和 __TRAIN__ 并行进行的, 即使将`test_initialization`设置为`false`也会进行一次__TEST__的数据 prefetch,  同样会进行`Transform`, 所以要注意相关的共享变量.
- BatchNorm 的 eps 默认为 1e-5, 这个数值是 切实 会对结果产生一定影响的, 在 absorb 参数时也要注意

## 过程记录
- 后向根据`top_diff`和前向结果算出各`blob`参数的`diff`, 以及`bottom`的`diff`, 所以分别对`blob`和`bottom`求导
- 传播时记得不同微分层乘上前面的梯度值`top_diff`,后传多个梯度值的话全部加起来
- `setup`是在加载网络时调用的, 加载完后不再调用

## 错误记录
1. Check failed: data_
  - 为 `blob shape` 错误, 一般是 `reshape` 函数出错, 也可能是网络设计错误导致 `shape` 传过来时负值错误

## 问题记录
1. caffe模型测试时`batch_norm`层的use_global_stats设为false居然没影响????  错觉
2. 训练过程开始良好, 中途出现后方部分卷积开始死亡(参数值非常低), 然后向前传染, 大部分卷积死亡, 表现为验证集上非常不稳定
   - 推测是ReLU死亡
3. caffe 和 opencv 一起 import 会出错
   - added `-Wl,-Bstatic -lprotobuf -Wl,-Bdynamic` to `LDFLAGS` and removed `protobuf` from `LIBRARIES` ( 参照 https://github.com/BVLC/caffe/issues/1917 )

## 犯2记录
1. `resize`层或者叫`upsample` `upscale` 层, 若训练时使用的缩放算法不同, 在卷积到比较小的时候(4x4)之类的, 会由于策略差异导致缩放前后误差非差大
2. test 或 upgrade 时 model 和 prototxt 写反
  > [libprotobuf ERROR google/protobuf/text_format.cc:274] Error parsing text-format caffe.NetParameter: 2:1: Invalid control characters encountered in text.  
.....   
*** Check failure stack trace: ***  
已放弃 (核心已转储)
3. 二分类问题 SoftmaxWithLoss 层不要设 ignore_label, ignore_label 是会忽略该 label 的 loss 和 diff 传递, 导致结果会完全倒向另一个 label , 因为 SoftmaxWithLoss 是计算准确率来算 loss 的


## 常见安装问题
1. 一般常见 protobuf 问题, 因为 Tensorflow 也用 protobuf, 不仅用, 还会自动升级 protobuf, 而 caffe 不怎么支持新版本的 protobuf, 所以如果配置了其他开源库的开发环境之后 caffe 报错了, 基本可以从几个方面检查 protobuf 有没问题.
   - `pip list`, 查看 protobuf 版本, 一般 2.6.1 比较通用, 如果是 3.5 那就换吧. 如果同时使用了 python2.7 和 python 3.5 的话那还要注意 pip 也分 pip2 和 pip3, 安装的库也分别独立. 可以在 `/usr/bin`, `/usr/local/bin`, `/$HOME/.local/bin` 下找到 pip 脚本, 打开就能看到它用的是 python2.7 还是 python3.5. ( _然后出现了下一个问题_ )
   - `protoc --version`, protobuf 依赖的东西, 查看它的版本和 protobuf 的是否一样, 不一样的话可以通过下载相应版本 release, 或者从源码安装 protobuf. 然后在 `/etc/ld.so.conf` 里面添加上一行 `/usr/local/lib`, 然后 `sudo ldconfig` 更新下链接库就行了. ( _然后出现了下一个问题_ )
   - `apt list | grep "protobuf"`, 有时候会有用 `apt-get install` 和 `pip install` 装了两种不同版本的 protobuf 的情况, 这时候可以 `apt` 删除并重新安装 protobuf ( _然后出现了下一个问题_ )
   - `File already exists in database: caffe.proto `, 库链接问题或者版本问题 ( 2.6.1 不好用 ), `pip uninstall protobuf` 删掉 protobuf, 重启, 加 -fPIC 到 configure, 然后 `./configure --disable-shared`, 然后在 protobuf 3.3 版本下 `cd $PROTOBUF_BUILD_DIR/python`, `python setup.py build`, `python setup.py test`, `python setup.py install` ( _然而出现了下一个问题_ )
     - 还可能是 caffe 玄学问题, 总之最简单的就是直接把能用的 caffe 替换过来
   - `make all` 时出现一堆 protobuf 未定义的引用问题. ( _未解, 回溯 2.6.1_ )
   - 2.6.1:
     - `caffe_pb2.py: syntax error`, 注释掉默认 caffe 的 `python/caffe/proto/caffe_pb2.py`, 至于为什么项目 caffe 没有用自己的 `caffe_pb2.py` 而用到默认 caffe, 是因为没有成功 `make pycaffe` ??? 总之应该是版本问题.
     - `File already exists in database: caffe.proto` 依旧存在这个问题, 在 `import caffe` 后 `import cv2` 会发生, 还是需要静态链接 protobuf, 这样可以解决:
       - > linking caffe against libprotobuf.a instead of libprotobuf.so could solve this issue
       - > I changed caffe's Makefile. Specifically, I added -Wl,-Bstatic -lprotobuf -Wl,-Bdynamic to LDFLAGS and removed protobuf from LIBRARIES.
         > I have uploaded my Makefile to gist (https://gist.github.com/tianzhi0549/773c8dbc383c0cb80e7b). You could check it out to see what changes I made (Line 172 and 369).

     - `File "/usr/lib/python2.7/dist-packages/caffe/pycaffe.py", line 13, in <module>
    from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver,
    libcaffe.so.1.0.0: cannot open shared object file: No such file or directory`. 这是 python 又喵了咪了用了默认 release 版 caffe, 删掉 `/usr/lib/python2.7/dist-packages/caffe`, 然后在工程头处 `import sys` 加`sys.path.insert('/home/sad/ENet/caffe-enet/python')` 和 `sys.path.insert('/home/sad/ENet/caffe-enet/python/caffe')` 再 `import caffe `, 问题终于解决!
    
2. `libcudnn.so.5: cannot open shared object file: No such file or directory`, ld 抽风, 需要专门刷新下 cuda 链接路径 :
```
sudo ldconfig /usr/local/cuda-8.0/lib64
```

3. `*** SIGSEGV (@0x100000049) received by PID 703 (TID 0x7f52cbb1c9c0) from PID 73; stack trace: ***` 或者 `Segmentation fault (core dumped)`, 可能是 python 层的使用出了问题
4. 段错误, `import caffe` 退出后错误, 有可能是用了 opencv contrib 的 `LIBRARY`, 在 `Makefile` 里删掉 `opencv_videoc` 什么的...

## 推荐安装方法

使用 CMake 来安装，推荐 ubuntu16.04 + gcc5.4 + python2.7 + CUDA8.0 + opencv3.4 + protobuf2.6 

实测 Ubuntu18.04 + gcc7 + python2.7 + CUDA10.2 + opencv3.4 + protobuf 3.11? 可以运行，但不支持 cudnn7.6.5。CUDA10.0 + cudnn7.3.1可以正常运作。

