# 个人网站框架

## 开发准备
必要环境
- 安装 nodejs
- 安装 pnpm

常用命令：
```bash
pnpm install  # 安装依赖
pnpm dev  # 本地开发调试
pnpm build  # 构建项目
```

## 功能介绍
框架 与 内容 分离

```bash
blog_framework
│          
├─.github
│  └─workflows
│          deploy.yml               # 启用 github action 之后，如果 仓库名 为 username.github.io，并且对 master 分支进行 push 之后，会自动执行构建项目、部署的流程
│          
├─public                            # 内容配置，以及文件
│  │  
│  └─custom
│      │  about.json                # 个人信息配置，如过想要添加键值对，请在 src/index.vue 读取完之后添加对应展示模块
│      │  blogs.json                # 博客列表配置，给出每个博客的基本信息，与内容路径，如果想要修改展示逻辑，修改 src/blogs.vue
│      │  news.json                 # 新事件配置，如果想要修改展示逻辑，修改 src/index.vue
│      │  researches.json           # 个人研究配置，如果想要修改展示逻辑，修改 src/researches.vue
│      │  
│      ├─blogs                      # 博客的具体文件
│      │  │  xxx.ipynb
│      │  │  xxx.md
│      │  │  
│      │  └─src
│      │      xxx
│      │              
│      └─images                     # paper 图片，个人主页图片等文件
│             xxxx.png
│              
└─src                               # 页面渲染逻辑
    │  App.vue, xx.ts               # 一些支持性文件
    │      
    ├─components
    │  │  paper.vue                 # 论文展示的组件，根据读取出的内容自动渲染
    │  │  
    │  └─layouts
    │          BaseHeader.vue       # 定义目录路由内容，如果想要添加内容，请将其与 src/pages 中的文件进行对应
    │      
    └─pages
           blogdetail.vue           # 根据 public/blogs.json 提供的路径读取博客内容，进行渲染，支持 .md，.ipynb 两种格式
           blogs.vue                # 读取 public/blogs.json 中的内容，完成 Blog 界面渲染
           index.vue                # 读取 public/about.json, public/news.json 中的内容，完成 Home 界面渲染
           researches.vue           # 读取 public/researches.json 中的内容，完成 Research 界面渲染
```
