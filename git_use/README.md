# 🛠️ Git 常用操作



##  1. 初始化与配置 
| 命令 | 说明 | 备注 |
| :--- | :--- | :--- |
| `git init` | 在当前目录初始化仓库 | 新项目第一步 |
| `git clone <url>` | 克隆远程仓库到本地 | 下载项目 |
| `git config --global user.name "Name"` | 设置用户名 | 首次配置 |
| `git config --global user.email "Email"` | 设置邮箱 | 首次配置 |

##  2. 日常开发流程 
| 命令 | 说明 | 场景          |
| :--- | :--- |:------------|
| `git status` | 查看当前状态 | 随时检查修改了哪些文件 |
| `git add .` | 添加所有修改到暂存区 | 准备提交        |
| `git add <file>` | 添加指定文件 | 只想提交部分文件    |
| `git commit -m "msg"` | 提交变动到本地仓库 | msg 描述改动    |
| `git log --oneline` | 以单行简洁模式查看历史 | 快速回顾历史      |

##  3. 同步远程仓库 
| 命令 | 说明 | 最佳实践 |
| :--- | :--- | :--- |
| `git pull` | 拉取远程代码并合并 | 开工前必做，防冲突 |
| `git push` | 推送代码到 GitHub | 功能完成或下班前 |
| `git remote -v` | 查看远程仓库地址 | 检查连接 |

##  4. 分支管理 (Branching)
| 命令 | 说明 | 备注 |
| :--- | :--- | :--- |
| `git branch` | 查看本地分支 | `*` 表示当前所在分支 |
| `git checkout -b <dev>` | 创建并切换到新分支 | 开发新功能/修 Bug 时用 |
| `git checkout <master>` | 切换回主分支 | 准备合并代码时 |
| `git merge <dev>` | 将 dev 分支合并到当前分支 | 通常在主分支执行 |

##  5. 撤销与暂存 
| 命令 | 说明 | 场景 |
| :--- | :--- | :--- |
| `git checkout -- <file>` | 丢弃工作区文件的修改 | 改乱了，想恢复原样 |
| `git reset --soft HEAD^` | 撤销上一次 commit (保留代码) | 提交过早或写错注释 |
| `git stash` | 暂存当前修改（不提交） | 突然要插队修 Bug |
| `git stash pop` | 恢复暂存的修改 | 插队任务完成，回来继续写 |