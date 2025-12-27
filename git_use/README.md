# ==========================================
# 🎵 MERT 项目 Git 操作速查手册
# ==========================================
# 第一步：查看当前修改了哪些文件 
git status
# 第二步：将所有修改加入暂存区 
git add .
# 第三步：将暂存区保存为本地存档
git commit -m "更新了训练脚本参数，优化了模型层数"
# 第四步：推送到 GitHub 
# 如果网络报错，执行下 [代理设置] 
git push

# ------------------------------------------
# 2. 撤回
# ------------------------------------------
# 写错了代码，还没 add，想全部还原 (慎用！会丢失修改)
git checkout .
# 或者 (新版命令)
git restore .

# 场景 B: 已经 git add 了，想撤回 add 操作 (变回红色状态)
git reset
# 或者
git restore --staged .

# 场景 C: 已经 git commit 了，想撤回 commit (保留代码，重新修改)
# HEAD~1 表示回退 1 个版本，--soft 表示保留代码
git reset --soft HEAD~1

# 场景 D: 已经 git push 了，发现错了，想再发一个“反悔补丁”抵消掉
# (这是最安全的远程撤回方式，不会搞乱别人的代码)
# 先用 git log 找到那个错误的 commit_id (一串长字符)
git revert <commit_id>


# ------------------------------------------
# 3. 网络代理 (Network Proxy)
# ------------------------------------------
# 开启代理 (上传报错 Failed to connect / Access restricted 时运行)
# 假设你的端口是 7890
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# 关闭代理 (平时不用梯子时运行，否则连不上)
git config --global --unset http.proxy
git config --global --unset https.proxy


# ------------------------------------------
# 4. 其他实用命令 (Utils)
# ------------------------------------------
# 查看提交历史
git log --oneline

# 创建并切换到新分支 (比如你要尝试改得面目全非，不想影响主线)
git checkout -b new-feature

# 切换回主分支
git checkout main

# 拉取远程最新的代码 (如果别人改了代码)
git pull