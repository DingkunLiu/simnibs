# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python 环境

必须使用 `conda` 虚拟环境。

```bash
eval "$(conda shell.bash hook)"
conda activate simnibs_env
```

- 执行任何 Python 命令前，先激活 `simnibs_env`

## 文件放置

- 新代码只生成在 `neuracle` 及其子目录中
- 生成代码前需要先输出方案，放在 `neuracle/docs` 下，格式为 Markdown
- 示例类运行代码放在 `neuracle/demo` 下
- 临时脚本和临时文件放在 `neuracle/private_gitignore` 下
- SimNIBS网格尺度参数验证文件在 `neuracle/mesh_validation`下
- 不要生成测试代码

## 代码风格

- 每个函数都要写 docstring
- docstring 使用 NumPy 风格
- docstring 中写清楚原理和用法
- 注释使用中文，专有名词使用英文
- 函数参数和返回值都需要标注类型
- 禁止在函数内部使用 import，所有 import 都放在模块头部
- 函数内部不要留空白行
- 函数内部变量名使用 `snake_case`

## 日志与提交

- `logger` 使用模块级变量，不要放到类内部
- 使用 `logger = logging.getLogger(__name__)`，不要写死模块名
- logging 使用 lazy `%` formatting
- 日志使用中文
- git message 使用中文，并写详细

## 远端运行

- 远端连接配置在 `.vscode/sftp.json` 中，需要连接远端时从这里读取信息，不要手写
- 远端执行 `neuracle` 代码前，同样先激活 `simnibs_env`
