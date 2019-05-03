---
title: nginx，mysql，docker，kubernetes相关博客读后笔记
tags: [总结, 基础]
---

这篇博客主要收集最近阅读的一些博客，主要围绕自己的项目和感兴趣的内容。

博客会按照主题进行分段，列出阅读的博客，然后列出其中自己总结的知识点和困惑的地方，然后进行扩展，博客目前包括但不限于以下内容，并持续更新中。。
- Nginx
- MySQL
- 分布式系统原理
- 微服务相关

# Nginx
## Nginx反向代理
背景：Nginx的使用动机是我在搭建博客的时候为了方便迁移，使用了docker。因此每提供一个服务都需要向外部暴露一个端口，同时每个服务还需要单独配置https服务。
所以我为了不在公网上暴露过多端口并统一配置，我使用了二级域名加上nginx的反向代理。

> 关于反向代理我阅读的博客是[Nginx负载均衡的详细配置及使用案例详解](https://www.cnblogs.com/wang-meng/p/5861174.html)

反向代理：Nginx的代理服务器使用upstream声明被代理的服务，并在server的location域配置相对应响应的域名和转发给的upstream，如下：
```
upstream proxy_server {
    # 声明upstream
    server 127.0.0.1:8080 weight=20;
    server 127.0.0.1.8081 weight=10;
} 

server {
    listen 80;
    server_name wuzequn.com;
    
    # 这样，就会将所有发送到http://wuzequn.com的服务就转发到本机两个服务端口上
    # 而且服务端口也不需要监听0.0.0.0，通过iptable或者防火墙服务设置只为代理服务器服务
    location / {
        proxy_pass http://proxy_server;
        root html;
        index index.html index.htm;
    }
}
```

keepalived: keepalived是常用来解决主从切换或者判断服务是否可用信息的服务，这篇博客主要解释了keepalived用于解决主备切换的原理。

keepalived使用的方法是vip(virtual ip address)来保证主机宕机之后，备机能够接替主机的功能，遵从的协议是VRRP协议，即路由器冗余协议。
- 主机除了在网关上记录一个实际ip(比如192.168.200.129)以外，还拥有一个虚拟ip(例如192.168.200.130)，并将这个虚拟ip绑定到路由器上，使得通过这个ip能访问到主机。
- 主机会定时向低于自己优先级的备机们发送VRRP包。
- 当备机(例如192.168.200.128)长时间没有收到VRRP包时，会在路由器内广播自己的VRRP包。
- 收集到每个备机的VRRP包之后，会比较与自己的优先级，如果当前最高优先级，则发ARP包广播，将vip(192.168.130)报告给网关绑定到自己身上。
- 这样，外部服务再通过vip访问服务的时候，路由器自然的就将包转发给备机，完成主备切换。


转发策略：上文提到了转发的策略，我通过阅读[转发策略](https://blog.csdn.net/hlg1995/article/details/84074749)和[重要算法](https://blog.csdn.net/liupeifeng3514/article/details/79018213)了解nginx的五种转发规则
- 轮询(逐个分配，剔除掉down的服务)
- 制定权重
- 按照访问者ip做hash分配主机，这样能解决session问题
- fair，按照响应时间快慢分配
- url的hash结果分配，有效利用服务器缓存


## nginx超时设置
关于上文判断节点是否down发散，向弄明白nginx是如何判断节点的状态，是不是也有心跳之类的机制，所以参考了[nginx中的超时设置，请求超时、响应等待超时等](https://www.cnblogs.com/lemon-flm/p/8352194.html)和[nginx upstream 容错机制](https://blog.csdn.net/qpfjalzm123/article/details/50008231)两篇文章。

可以总结为，Nginx除了proxy_next_upstream选项会参考返回码以外，只以connect refuse和timeout状态作为节点状态的判断依据，在与节点有关的超时配置里，有以下三点：
- proxy_connect_timeout :后端服务器连接的超时时间_发起握手等候响应超时时间
- proxy_read_timeout:连接成功后_等候后端服务器响应时间_其实已经进入后端的排队之中等候处理（也可以说是后端服务器处理请求的时间）
- proxy_send_timeout :后端服务器数据回传时间_就是在规定时间之内后端服务器必须传完所有的数据

所以如果连接握手超时，则会refuse判断节点down掉，如果后两个超时，则会影响转发策略。同时，如果一个节点频繁返回500-504，也会在配置proxy_next_upstream的时候影响节点状态的判断。

## keepalived简单入门
从上面主从切换做发散，阅读了[Keepalived学习总结](https://blog.51cto.com/xuweitao/1953167)简单了解了keepalived的原理。主要可以分成以下五个部分，主要是core，check，vrrp：
- core，是keepalived的核心，复杂主进程的启动和维护，全局配置文件的加载解析等
- check，负责healthchecker(健康检查)，包括了各种健康检查方式，以及对应的配置的解析包括LVS的配置解析
- vrrp，VRRPD子进程，VRRPD子进程就是来实现VRRP协议的
- libipfwc，iptables(ipchains)库，配置LVS会用到
- libipvs*，配置LVS会用到

![](keepalived.gif)

从上面可以看到，keepalived主要就是用来处理主从切换的场景。


## Nginx IO模型
在看完以上内容之后，简单对Nginx能够支撑高并发的原理进行了解和总结，主要针对Nginx的IO模型，主要参考的文章是[NGINX的IO模型详解](https://www.cnblogs.com/LuckWJL/p/9884041.html)，主要的内容参考博客就能理解，这里只做要点的总结和归纳，便于复习和回顾。

### Linux基础
用户空间和内核空间：由于用户进程无法直接操作硬件，所以例如读取IO等操作，必须透过系统调用，通过CPU处理对应的中断并接收才能实现。这个过程就需要进行进程切换，所以优化的要点往往是尽量减少进程的切换。

文件描述符和缓Linux IO：Linux往往会将所有的设备资源都按照文件形式进行管理，如果接收到一个IO操作，他首先会将这部分数据放到文件系统页缓存里，但是这部分缓存属于内核态的取址空间，想要用户态程序访问到，必须将它复制到用户空间的缓存区。所谓Socket做的事情就是这样，进行系统调用，等待内核缓冲区数据准备好，然后复制到自己的地址空间。

### 同步/异步
同步和异步是客户这一端处理逻辑的概念。所谓同步，是指客户端发起一个请求，在服务端给他返回数据或者超时之前，一直处于等待状态，不会做任何操作。而异步逻辑是指，在发起一个请求之后，不要求立即获得结果，先处理不依赖这部分结果的那部分逻辑，等服务端返回结果后，再通过回调的方式，接着处理之后的逻辑。

### 阻塞/非阻塞/多路复用IO
阻塞场景一般发生在一定要等到结果否则无法进行下一步动作的场景，例如Socket要获取到内核空间获取到的数据才能复制一样。所以阻塞场景常常发生在服务端。通常可以分成以下几种方式：
- 阻塞：在结果返回之前，进程进入非执行状态，进程只能自己通过系统调用将自己挂起，所以阻塞状态只能自己发起，这时候系统获得这个进程的进程指针并保留PCB信息，等到系统完成了对应结果的状态（例如内核空间缓冲区放入数据），就执行进程指针，使阻塞进程进入执行状态，将数据复制到用户空间。
- 非阻塞状态：非阻塞状态就会采用循环的方式一直轮询系统的状态，一旦轮询发现已经ready，就执行接下来的操作，但是这种情况会导致CPU空转，影响性能。
- 多路复用IO：多路复用IO本质上也是一个阻塞IO，但是他可以同时阻塞多个IO操作（例如端口或者文件描述符），而且同时对多个读操作，多个写操作进行检测（不一定是端口），直到有数据可读或可写时，才真正调用I/O操作函数。

所以Nginx采用的内存模型使用的是多路复用IO模型，有select，poll，epoll等多种，这些都是系统调用方式，所以他在编译时，会采用平台支持的最高效的模型进行编译。

## SELECT/POLL/EPOLL
> 上面了解了多路复用IO技术，就再往深一步，阅读[深入理解SELECT、POLL和EPOLL](https://wyj.shiwuliang.com/%E6%B7%B1%E5%85%A5%E7%90%86%E8%A7%A3SELECT%E3%80%81POLL%E5%92%8CEPOLL%20.html)和[Epoll的简介及原理](https://zhuanlan.zhihu.com/p/56486633)

### select,poll,epoll各自的原理和问题
select，poll，epoll都是多路IO复用的实现，意思是一个操作里同时监控多个输入输出数据源，当其中一个可以返回游泳数据的时候，对其进行读写操作。

select是一个系统调用，当一个可读写事件发生的时候，他会遍历所有fd，并返回fd是不是pollable的，每次遍历过程，select都需要将全部fd从用户空间复制到内核空间，当有可读写事件或超时的时候，就会安排调度器唤醒对应的进程。

poll解决了上面的一个问题，因为每次都要遍历所有fd，所以select规定只有1024-2048大小，所以poll改变了select的集合描述方式，使得大小可以大于1024。

epoll有两个突破：第一个是将复制fd的过程省略，直接使用共享的一块内核空间进行fd的增删改查，用红黑树进行组织，事件比较短。另外，每次只返回可读写的fd，并返回。具体而言，实现方式是epoll_create会创建一个之前没有创建的fd，epoll_ctl会建立在该fd上的singel_poll_wait_list上，等到fd可读写了，就会被加到ready_list上去，然后遍历ready_list上面的wait_entry_sk对应的singel_poll_wait_list的进程，恢复进程并从epoll_wait返回。

关于水平触发和边缘触发，我还没有完全理解，之后补充。。。。
