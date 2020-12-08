### 搭建一个属于自己的 Jekyll 博客

Jekyll 是 Github Page 推荐的静态网页、博客生成器，他搭配 Github Page 可以快速搭建出一个像模像样的博客，并且网络上提供了很多优美的免费主题，也有[官方推荐的主题网址](https://jekyllrb.com/docs/themes/)，可以直接拿来使用，非常方便。但是在我使用了一段时间网上的主题后，有一种“知其然而不知其所以然“的感觉，虽然用起来方便，但是并不知道那些各种各样的文件和文件夹都有什么作用，如果想添加或修改一些功能，也只能按照博客上的方法一步一步操作，并不知道其背后的原理。

于是我想要寻找有没有可以不使用别人提供的主题，从头建立一个自己的博客的方法，在我浏览 Jekyll 的官网的时候，我发现官方有一个很详细的教程，就是在引导我们如何自己构建一个简单的博客，所以我决定按照官网的指南一步一步的构建，作此博客一方面记录自己的完成过程，一方面也可以作为一个简单的教程，可以供大家参考。官方教程：https://jekyllrb.com/docs/step-by-step/01-setup/

在这篇博客中，你可以学到：

+ Jekyll 构建的文件夹下各个文件的内容和功能
+ 一些简单的 Liquid 语言
+ 

#### 准备工作

在 Ubuntu 上搭建一个属于自己的 Jekyll 博客，我们需要首先安装 Jekyll 和 Bundler，这部分工作在网络上有很多的资源，推荐参考[官网](https://jekyllrb.com/docs/installation/) 的下载和安装教程。

如果你已经安装了 Jekyll 和 Bundler 那我们就继续。

#### 创建一个 Jekyll  站点

在创建网站之前，我们先熟悉下接下来我们要用到的 Jekyll 命令行指令：

+ `jekyll new PATH` 在目录 PATH 下创建一个新的 Jekyll 网站，并附带一个默认的基于 gem 的主题

+ `jekyll new PATH --blank` 在目录 PATH 下创建一个空白的 Jekyll 网站
+ `jekyll serve` 在本地生成网站，我们可以通过 http://127.0.0.1:4000/ 去访问他，这条指令方便我们在本地查看目前网站是什么样子的。

在了解了这些指令后，我们就开始创建一个新的 Jekyll 网站吧，为了使创建的文件更加简介，我们选择创建一个 blank site：

```powershell
# 创建在 myTheme 目录下创建
$ jekyll  new  myTheme --blank
#输出：New jekyll site installed in PATH/myTheme.
```

#### 生成网站并打开

```powershell
$ cd myTheme
$ jekyll serve
# 输出：
# 	Configuration file: PATH/myTheme/_config.yml
#           		     Source: PATH/myTheme
#       	   Destination: PATH/myTheme/_site
# Incremental build: disabled. Enable with --incremental
#     	Generating... 
#       	done in 0.025 seconds.
# Auto-regeneration: enabled for 'PATH/mytheme_blank'
#        Server address: http://127.0.0.1:4000/
#  Server running... press ctrl-c to stop.
```

这样就让一个网页成功的在本体创建了，点击输出中的网址 http://127.0.0.1:4000/，就可以访问网页。一个 blank 的网页应该如下图所示：

![blank](/home/hezhuohao/ZhuohaoHe.github.io/img/in-post/2020-11-24-Step-by-Step-Tutorial-of-Jeklly/blank.png)

观察刚刚的输出中的三行，

```powershell
# 	Configuration file: PATH/myTheme/_config.yml
#           		     Source: PATH/myTheme
#       	   Destination: PATH/myTheme/_site
```

可以发现：

+ _config.yml 文件是网站的配置文件，在网站生成前会先读取这个文件中的内容
+ 构建网站的资源是来自于 myTheme 文件夹
+ 构建后的网站被放置在了 myTheme/_site 文件夹下

#### 总览文件夹的文件结构

```powershell
.
├── assets
│   └── css
│       └── main.scss
├── _config.yml
├── _data
├── _drafts
├── _includes
├── index.md
├── _layouts
│   └── default.html
├── _posts
├── _sass
│   └── main.scss
└── _site
    ├── assets
    │   └── css
    │       ├── main.css
    │       └── main.css.map
    └── index.html

```

这是我们目前文件夹下的文件结构（如果没有构建过网址，则没有 `_site` 文件夹），看起来有些复杂，可以先简化一下，下面的文件结构就是简化后的，他只展示了基础的文件和文件夹。注释简单标注了这些文件或文件夹的功能，并不需要完全理解，只需要大概了解一下文件夹中最基础的文件和文件夹就可以，接下来会详细介绍。

```powershell
.
├── _config.yml					# 网站的配置文件，配置一些基本信息
├── _drafts								# 草稿文件夹
├── _includes						# 存放引用文件 
├── index.md						# 主界面的内容
├── _layouts						# 样式文件夹，存放各种样式
│   └── default.html
├── _posts							# post 文件夹
└── _site								# 存放网站构建后
```

#### 查看 index.md

