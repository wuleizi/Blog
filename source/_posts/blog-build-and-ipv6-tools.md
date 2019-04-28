---
title: 个人博客（hexo next）日志及校外ipv6 bt访问工具
tags: 运维
date: 2018-04-27 19:00:20
---


博客就是记录最近的工作，所以就把这个博客的搭建过程以及校外访问bt小工具的日志作为个人博客的第一篇。
<!-- more -->

## 博客搭建过程
### 调研
[直接看步骤](#hexo_tutor)

我在知乎上翻阅了大部分关于博客工具的回答，基本了解个人博客一般采用脚本语言搭建Web（比如Django等）和静态网页工具两种。由于不想增加额外的运维成本，基本定位到静态网页工具上。由于不想在前端模板方面投入太多精力，所以基本去找有比较稳定的主题的工具。参考[准备自己建一个个人博客，有什么好的框架推荐？ - 知乎](https://www.zhihu.com/question/24179143/answer/33734461)和一些其他的文章，基本把调研内容锁定在jekyll和hexo两种工具上。

[Jekyll](https://jekyllrb.com)是基于Ruby脚本实现的博客生成工具，是经常与Github Pages配合使用的工具。目前比较常用的主题是[huxpro](https://github.com/Huxpro/huxblog-boilerplate)，我之前的同事[王喆的博客](http://wangzhe.website/)也是基于这个主题构建的。但是，经过试验发现，主题代码已经更新较多，相关文档却没有对应更新，一些设置找不到对应文档只能去查看源码，所以测试了Hexo之后选取了Hexo。但是，毕竟Jekyll是Github配套工具，相关的主题也比较完整，所以想要自己折腾的可以多尝试尝试。


[Hexo](https://hexo.io/)是基于Node.js脚本实现的博客生成类工具，如果你之前做过有关node的工作，相对接触Hexo也比较简单。Hexo比较流行的主题是[Next](http://theme-next.iissnan.com/)，功能没有那么丰富，所以配置相对简单而且全面，文档十分详细，运维成本低。而且Hexo的官网还有对应的youtube视频可以看，懒得看的可以对着视频做也不会出错。

详细教程可以在[Hexo官网](https://hexo.io/docs/)查看，以下给出简单步骤。

### <span id=hexo_tutor>步骤</span>
安装Node.js，windows的可以[下载](https://nodejs.org/en/)，其余同学可以用Node Version Manager安装
```
# 安装nvm
$ wget -qO- https://raw.githubusercontent.com/creationix/nvm/v0.33.2/install.sh | bash
# 安装Node.js
$ nvm install stable
```
安装git可以参考 [git](https://git-scm.com/downloads)

#### 准备工作
如果想要查看详细的说明可以查看 [next](http://theme-next.iissnan.com/getting-started.html)

安装Hexo
```
$ npm install -g hexo-cli
```

博客初始化
```
$ hexo init <folder>
$ cd <folder>
$ npm install
```
下载主题
```
$ cd <folder>
$ git clone https://github.com/iissnan/hexo-theme-next themes/next
```
启用主题，编辑<code>_config.yml</code>，找到theme字段，改为next

```
theme: next
```
验证主题 <code>hexo s --debug</code>，显示以下内容，访问<code>http://localhost:4000</code>如果发现未生效，清除Hexo静态文件<code>hexo clean</code>再执行，以后的步骤任意改动类似。

```
INFO  Hexo is running at http://0.0.0.0:4000/. Press Ctrl+C to stop.
```
![theme-example](http://theme-next.iissnan.com/uploads/five-minutes-setup/validation-default-scheme-mac.png)

#### 配置
选择设计效果Scheme，修改<code>themes\next\_config.yml</code>，我采用的主题版本实现了四种效果，选取了Gemini。
```
# Schemes
#scheme: Muse
#scheme: Mist
#scheme: Pisces
scheme: Gemini
```
设置中文<code>_config.yml</code>

```
language: zh-Hans
```
<span id="menu">设置菜单<code>themes\next\_config.yml</code>，Next使用的是[Font Awesome](http://fontawesome.io/)，如果你想要的菜单图标没有显示，可以在网站上寻找理想的图标并在<code> ||</code>后修改。如果想要icon生效，可以将<code>menu_icons</code>的<code>enable</code>设置为*true*</span>

```
menu:
  home: / || home
  about: /About/ || user
  tags: /tags/ || tags
  categories: /categories/ || th
  archives: /archives/ || archive
  #schedule: /schedule/ || calendar
  #sitemap: /sitemap.xml || sitemap
  #commonweal: /404/ || heartbeat
```
设置头像<code>themes\next\_config.yml</code>，可以选取互联网地址或者本地地址
```
avatar: http://www.wuzequn.com/images/touxiang.png
```

设置代码高亮<code>themes\next\_config.yml</code>，next使用的是[Tomorrow Theme](https://github.com/chriskempson/tomorrow-theme)，可以在链接中查看高亮效果。
```
# Code Highlight theme
# Available value:
#    normal | night | night eighties | night blue | night bright
# https://github.com/chriskempson/tomorrow-theme
highlight_theme: night
```

设置侧边栏社交链接<code>themes\next\_config.yml</code>，图标的设置和[菜单](#menu)类似。
```
# Social Links.
# Usage: `Key: permalink || icon`
# Key is the link label showing to end users.
# Value before `||` delimeter is the target permalink.
# Value after `||` delimeter is the name of FontAwesome icon. If icon (with or without delimeter) is not specified, globe icon will be loaded.
social:
  #GitHub: https://github.com/yourname || github
  邮件: mailto:wuzqbupt@gmail.com || envelope
  知乎: https://www.zhihu.com/people/wu-ze-qun || globe
  微博: https://weibo.com/u/1922768971 || weibo
  领英: https://www.linkedin.com/in/zequn-wu-038a5b133/ || linkedin
```

网页访问量统计<code>themes\next\_config.yml</code>，代码是基于[不蒜子统计](http://busuanzi.ibruce.info/)实现的。
```
# Show PV/UV of the website/page with busuanzi.
# Get more information on http://ibruce.info/2015/04/04/busuanzi/
busuanzi_count:
  # count values only if the other configs are false
  enable: true
  # custom uv span for the whole site
  site_uv: true
  site_uv_header: <i class="fa fa-user"></i>
  site_uv_footer: 
  # custom pv span for the whole site
  site_pv: true
  site_pv_header: <i class="fa fa-eye"></i>
  site_pv_footer: 
  # custom pv span for one page only
  page_pv: true
  page_pv_header: <i class="fa fa-file-o"></i>
  page_pv_footer: 
```

本地搜索
* 安装<code>hexo-generator-searchdb</code>，在站点的根目录下执行以下命令：
```
$ npm install hexo-generator-searchdb --save
```
* 配置<code>_config.yml</code>，增加以下内容：
```
search:
  path: search.xml
  field: post
  format: html
  limit: 10000
```
* 配置<code>themes\next\_config.yml</code>，启用本地搜索功能
```
search:
  path: search.xml
  field: post
  format: html
  limit: 10000
```


其他类似于昵称和站点描述等信息可以在<code>_config.yml</code>里修改。

#### 内容编辑
Hexo的内容都可以用[Markdown](https://zh.wikipedia.org/zh-hans/Markdown)，具体的格式可以参考[教程](http://www.markdown.cn/)

##### about页面
首先生成about页面，首先在站点的根目录下执行以下命令：
```
$ hexo new page About
```
然后会在<code>source</code>目录下生成一个About文件夹，然后编辑<code>source/About/index.md</code>介绍你自己。

##### tags页面
首先生成about页面，首先在站点的根目录下执行以下命令：
```
$ hexo new page tags
```
然后会在<code>source</code>目录下生成一个tags文件夹，然后编辑<code>source/tags/index.md</code>，修改为:
```
---
title: tags
date: 20xx-xx-xx xx:xx:xx
type: "tags"
comments: false
---
```
##### categories页面
首先生成categories页面，首先在站点的根目录下执行以下命令：
```
$ hexo new page categories
```
然后会在<code>source</code>目录下生成一个categories文件夹，然后编辑<code>source/categories/index.md</code>，修改为:
```
---
title: categories
date: 20xx-xx-xx xx:xx:xx
type: "categories"
comments: false
---
```

##### 内容操作
hexo有三种内容类型，分别文<code>page</code>，<code>post</code>，<code>draft</code>。可以用以下内容新建：
```
$ hexo new [page|post|code] <file-name>
```

<code>page</code>就是基础页面，你可以在<code>http;//127.0.0.1/</code>后添加页面名称访问到对应文件夹下内容。

<code>post</code>新建到<code>source</code>目录下，名称就是文件名称。

如果不想发布未编辑完成的草稿，可以新建<code>draft</code>，待编辑完成，发布草稿。
```
$ hexo publish <draft-name>
```

##### 发布网站
Hexo是用脚本将*Markdown*文件生成静态*html*页面的工具，用一下命令生成一个<code>public</code>静态文件夹。
```
$ hexo generate
```
同时，Hexo也提供了一个服务器用于显示内容。ye也可以加<code>-p</code>选项配置http端口，默认为4000
```
$ hexo server
```
如果发现你的修改没有生效，可以执行以下命令清楚数据文件和<code>public</code>文件夹，之后再重新生成。
```
$ hexo clean
```
另外，hexo也提供了将静态文件夹部署的功能，我使用了[码市](http://coding.net/)作为git仓库网站，具体配置git方法可以[参考](https://www.cnblogs.com/gdfhp/p/5889168.html)，本地需要配置<code>_config.yml</code>并安装[hexo-deployer-git](https://github.com/hexojs/hexo-deployer-git)，更多部署方式可以[参考](https://hexo.io/docs/deployment.html)
```
# Deployment
## Docs: https://hexo.io/docs/deployment.html
deploy:
  type: git
  repo: https://<git-address>
  branch: <branch-name>
```
执行以下命令，就可以将<code>public</code>中的静态文件上传到git仓库。
```
$ hexo deploy
```

如果你采用Github.io作为展示方案，参考[教程](https://blog.csdn.net/walkerhau/article/details/77394659?utm_source=debugrun&utm_medium=referral)，执行完部署步骤，就可以将静态文件部署到Github.io上。

##### 其他修改操作
* [hexo插入本地图片资源](https://blog.csdn.net/sugar_rainbow/article/details/57415705)
* 插入公式：在博客的添加<code>mathjax: true</code>
* [常用的Markdown数学公式](https://blog.csdn.net/zdk930519/article/details/54137476)
* [Markdown插入Jupyter Notebook](https://www.jianshu.com/p/6c1196f12302)


## vps
### vps购买
由于之后还需要搭建科学上网小工具和ipv6工具，所以在选取vps运营商的时候看重有ipv6出口，所以根据同学建议选择了<span id="vps">搬瓦工OpenVZ</span>的云主机。

之后又调研了vps运营商，发现还有AplhaRacks的7美元方案，对于不注重运行速度的童鞋可以采用这个方案。

同时推荐[直呼过瘾](https://www.zhihu.in/)，查询性价比比较高的方案，但是要注意如果要搭建ipv6小工具，需要注意vps中有ipv6通道。

### 博客配置
<code>环境：CentOS 7 64 bit</code>

博客采用的是Nginx作为Web服务器，安装Nginx：
```
$ sudo yum install epel-release
$ sudo yum install nginx
```
从git拉取代码并授权
```
$ git clone https://<your-repo-address>
$ cd <git-dir>
$ sudo chmod -R 777 .
```
配置Nginx，将Nginx访问的根目录配置为本地仓库的根目录。修改<code>/etc/nginx/nginx.conf</code>中的内容。
```
    server {
        listen       80 default_server;
        listen       [::]:80 default_server;
        server_name  _;
        # root         /usr/share/nginx/html;
        root <your-repo-local-path>;
        # Load configuration files for the default server block.
        include /etc/nginx/default.d/*.conf;

        location / {
                index index.html;
                autoindex on;
        }
        error_page 404 /404.html;
            location = /40x.html {
        }

        error_page 500 502 503 504 /50x.html;
            location = /50x.html {
        }

    }


```

### https配置
CentOS可以参考[DigitalOcean的解决方案](
https://www.digitalocean.com/community/tutorials/how-to-create-a-self-signed-ssl-certificate-for-nginx-on-centos-7)。


但是，配置了<code>HTTPS</code>之后，如果不购买CA证书，Chrome就会在网站显示<code>Not Secure</code>。如果觉得不需要购买，可以取消<code>HTTPS</code>。

```
# 失效SSL
$ mv /etc/nginx/conf.d/ssl.conf /etc/nginx/conf.d/ssl.conf.bak 
# 失效redirect
$ mv /etc/nginx/default.d/ssl-redirect.conf /etc/nginx/default.d/ssl-redirect.conf.bak
# 检测配置文件格式
$ nginx -t
# 是配置生效
$ nginx -s reload
```
此时，访问就会恢复<code>HTTP</code>。如果发现未生效，[清除Chrome缓存数据](https://jingyan.baidu.com/article/fea4511a2d207ff7bb91252a.html)。

> 2018-09-24 更新
可以使用免费CA证书，我使用的是[Let's Encrypt](https://github.com/certbot/certbot)，步骤如下:

```
# 准备工作 python环境

# 检查系统是否安装git,如果已经自带有git会出现git版本号，没有则需要我们自己安装
git  --version 

# git 安装
yum install git

# 检查Python的版本是否在2.7以上
python -v //2.6版本

# 安装python所需的包
yum install zlib-devel
yum install bzip2-devel
yum install openssl-devel
yum install ncurses-devel
yum install sqlite-devel



# 获取letsencrypt
git clone https://github.com/letsencrypt/letsencrypt

# 进入letsencrypt目录
cd letsencrypt

# 生成证书 -nginx为例
./certbot --nginx certonly
# 然后输入你的邮箱和所要添加https的域名
# 生成秘钥完毕
```

上文DigitalOcean的解决方案中提到的nginx配置文件的秘钥路径：
```
#打开linux配置文件，找到HTTPS 443端口配置的server
 ssl_certificate /etc/letsencrypt/live/域名/fullchain.pem;
 ssl_certificate_key /etc/letsencrypt/live/域名/privkey.pem;
```

修改http配置，将http转成https服务：
```
server {  
      listen      80;  
      server_name    域名;  
      return      301 https://域名$request_uri;  
    }
```


此时，CA证书90天会过期，过期所以需要续签：
```
./letsencrypt-auto renew --force-renewal
```


最后你可以通过以下网址检测https服务
```
https://www.ssllabs.com/ssltest/analyze.html?d=域名
```


## 域名注册
域名注册商很多，国内需要实名注册，要求的内容较多且周期比较长。国外的注册商通常自己提供DNS服务器，对速度有一定的影响。

如果不是米商，个人博客通常还是选取性价比高的注册商。域名的购买需要注意两个要点：
* 注册价格，可以使用[比价网站](https://www.domcomp.com/)。
* 注意续租价格，一般<code>.com</code>的网站每年价格相同。

注册成功后将云主机ip绑定，绑定DNS可以参考[百度经验](https://jingyan.baidu.com/article/6079ad0e6b95f428ff86db08.html)。

## 校外访问bt ipv6小工具
### 服务器端
上网工具其实就是将本地无法访问的请求发送给可以访问的服务器，服务器取回数据后再转发给客户端。如[上文](#vps)，服务器采用的是搬瓦工的VPS，所以首先要在控制面板上开启ipv6地址。以root安装ss服务。
```
wget --no-check-certificate -O shadowsocks-go.sh https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks-go.sh
chmod +x shadowsocks-go.sh
./shadowsocks-go.sh 2>&1 | tee shadowsocks-go.log
```
安装中会选择转发的端口和密码。
安装完成后，会提示：
```
Congratulations, Shadowsocks-go server install completed!
Your Server IP        :your_server_ip
Your Server Port      :your_server_port
Your Password         :your_password
Your Encryption Method:your_encryption_method

Welcome to visit:https://teddysun.com/392.html
Enjoy it!
```
如果安装成功，ss服务会开机自动启动。

如果想卸载，可以执行：
```
./shadowsocks-go.sh uninstall
```
如果想要为多用户并行服务，可配置多转发端口，编辑<code>/etc/shadowsocks/config.json</code>:
```
{
    "server":"0.0.0.0",
    "port_password":{
         "8989":"password0",
         "9001":"password1",
         "9002":"password2",
         "9003":"password3",
         "9004":"password4"
    },
    "method":"aes-256-cfb",
    "timeout":600
}

```

配置成功之后重启服务
```
/etc/init.d/shadowsocks start
/etc/init.d/shadowsocks stop
/etc/init.d/shadowsocks restart
/etc/init.d/shadowsocks status
```

> 2019-04-28更新
另外，因为OVZ更新了，所以如果你使用的是KVM架构，也可以参考[搬瓦工](https://www.bandwagonhost.net/2144.html)这篇文章用渠道技术搞定IPV6


### 客户端
[下载客户端](https://shadowsocks.org/en/download/clients.html)，然后根据服务器的配置，编辑端口和密码。同时，shadowsocks客户端可以设置PAC规则，节省转发流量，也可以在PAC文件中增加转发网址。设置好后可以访问ipv6和科学上网，可以[北邮人bt](http://bt.byr.cn)测试。
![bt](https://zhbbupt.github.io/images/bt.PNG)

### 设置utorrent
* 打开设置 -> 连接
* 代理服务选择socks5，代理127.0.0.1，端口1080，勾选通过代理服务器解析主机名和 对于点对点连接使用代理服务器
![ut-setting](https://zhbbupt.github.io/images/utorrent.PNG)
* 如果设置的系统代理方式为<code>PAC代理</code>，需要在PAC文件中加入<code>.byr,cn</code>。