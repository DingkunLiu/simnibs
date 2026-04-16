# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python 环境

**必须使用 conda 虚拟环境**。

### 激活环境

```bash
eval "$(conda shell.bash hook)"
conda activate simnibs_env
```

### 生成时的规范

* 执行python命令前必须激活虚拟环境
* 新代码都只生成在`neuracle`以及子文件夹中
* 生成代码前需要输出方案，放在`neuracle/docs`下，格式为markdown
* 例子类的运行代码都放在`neuracle/demo`下
* 临时用的脚本和文件放在`neuracle/private_gitignore`下
* 不要生成测试代码
* 每个函数都要写docstring
* docstring使用NumPy格式
* 在docstring中要写清楚原理和用法
* 注释使用中文，专有名词使用英文
* 函数参数需要标注类型，返回值也需要标注类型
* 禁止在函数内部使用import，所有import都在模块头部
* 函数内部不要留空白行
* logger用模块级别的,不要放到类内部
* 用logger = logging.getLogger(__name__)的方式，不要写具体的模块名字
* Use lazy % formatting in logging functions
* 函数内部变量名使用snake_case
* 日志都使用中文
* git message用中文，写详细
