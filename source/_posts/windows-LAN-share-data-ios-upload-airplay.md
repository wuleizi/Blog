---
title: Windows局域网与手机共享文件以及iOS文件操作
date: 2018-05-16 12:16:17
tags: [运维, 生活]
---
### 相关背景
由于最近拍的照片比较多，有一些需要传到电脑里进行备份。还有，有将电脑里的文件用手机投屏到电视上或投影仪上的需求。因此，有网络硬盘做存储中心，手机做控制中心的需求。

我只是个人进行文件备份，对延迟没有突出需求，所以决定将电脑作为局域网中的NAS。

### Windows做文件共享
此步骤参考了[百度经验](https://jingyan.baidu.com/album/fec7a1e53efe621190b4e7ae.html)
* 右击桌面网络----属性----更改高级共享设置 （注释：查看当前网络 比如：家庭网络、公共网络 等!) "我这里为公共网络"
![](win1.png)
* 选择 公共网络---选择以下选项：启动网络发现------启动文件和打印机共享-----启用共享以便可以访问网络的用户可以读取和写入公用文件夹中的文件(可以不选）----关闭密码保护共享( 注释：其他选项默认即可!)
![](win2.png)
* 保存！
![](win3.png)
* 选择需要共享的文件夹 (比如：DY) 右击--属性
![](win4.png)
* 共享---- 选择 共享（S）...---弹出对话框---添加“Guest”（注释：选择“Guest”是为了降低权限，以方便于所有用户都能访问！）---共享
![](win5.png)
* 选择 高级共享... ----选择 共享此文件 ----确定！
![](win6.png)
* 其他用户，通过开始---运行---\\IP （快捷键 WIN+R）\\IP 访问你共享的文件!
![](win7.png)
* 共享成功!
![](win8.png)
### ios共享数据
下载[FileExplorer](https://itunes.apple.com/cn/app/fileexplorer/id499470113?mt=8)
本教程参考自[iOS神器FileExplorer精简使用教程](https://zhuanlan.zhihu.com/p/26745885)
具体操作方法为：
* 保持 Windows 和 iPad 在同一局域网环境下，如：使用同一个WiFi。
![](ios1.jpg)
* 点击右上角“+号”，选择Windows
![](ios2.jpg)
* 往下拉，选择网络邻居里的项目（如图即：DESKTOP-71EDJSC），并选择“注册的用户”\
![](ios3.jpg)
* 输入Windows的账户名和密码，然后，尽情浏览你的共享文件吧
![](ios4.jpg)
* win10：遇到权限不足时，点击设置-帐户-你的电子邮件和帐户-改用本地帐户登录，使用此时的帐号密码登录 FileExplorer 即可。

> 其余操作可以到[FE官方文档查看](https://www.skyjos.com/fileexplorer/help/help_main.php)

### ios文件上传
#### 照片与文件上传
具体操作查看[FE文件传输](https://www.skyjos.com/fileexplorer/help/help_main.php?section=smb_transfer)
#### ios11文件格式问题
ios11之后，iphone的照片格式变成了<code>.heic</code>，所以windows也无法查看与转换，有两种转换方法：
* 单独上传可以用ios自带的 *文件* 进行转换
![](pro1.PNG)
![](pro2.PNG)
![](pro3.PNG)
然后将icloud里面的文件复制到windows文件夹中。
![](pro4.PNG)
* 批量上传可以使用批量转换工具，我使用的是开源的转换工具[HEIF Utility](https://liuziangexit.com/HEIF-Utility/)，github地址为[HEIF Utility](https://github.com/liuziangexit/HEIF-Utility)

### 共享文件投屏
具体操作方法查看[FE投屏方法](https://www.skyjos.com/fileexplorer/help/help_main.php?section=airplay)。

> 注意这里需要你的投放设备支持DLNA协议的设备，如果搜不到airplay设备，可以给你的设备下载投屏工具，类似于[乐播投屏](http://www.hpplay.com.cn/)。

### iphone通过airplay音频输出到windows

可以使用[shairport4w](http://www.lightweightdream.com/sp4w/shairport4w/?lang=en)
