# codes/scan_dependencies.py
"""
扫描所有开源任务的专家实现代码，提取第三方库导入语句，
汇总为去重排序的依赖清单。

支持两种读取途径：
1. 从基准数据集CSV的CodeString字段提取
2. 从各任务目录下的.py文件读取

两种来源内容等价，默认使用CSV以避免文件系统遍历。
"""

import ast
import re
import sys
from pathlib import Path

import pandas as pd


# Python标准库模块名（3.10+常见），用于过滤
STDLIB_MODULES = {
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio',
    'asyncore', 'atexit', 'base64', 'bdb', 'binascii', 'binhex',
    'bisect', 'builtins', 'bz2', 'calendar', 'cgi', 'cgitb', 'chunk',
    'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections',
    'colorsys', 'compileall', 'concurrent', 'configparser', 'contextlib',
    'contextvars', 'copy', 'copyreg', 'cProfile', 'crypt', 'csv',
    'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm', 'decimal',
    'difflib', 'dis', 'distutils', 'doctest', 'email', 'encodings',
    'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput',
    'fnmatch', 'fractions', 'ftplib', 'functools', 'gc', 'getopt',
    'getpass', 'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib',
    'heapq', 'hmac', 'html', 'http', 'idlelib', 'imaplib', 'imghdr',
    'imp', 'importlib', 'inspect', 'io', 'ipaddress', 'itertools',
    'json', 'keyword', 'lib2to3', 'linecache', 'locale', 'logging',
    'lzma', 'mailbox', 'mailcap', 'marshal', 'math', 'mimetypes',
    'mmap', 'modulefinder', 'multiprocessing', 'netrc', 'nis', 'nntplib',
    'numbers', 'operator', 'optparse', 'os', 'ossaudiodev', 'pathlib',
    'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform',
    'plistlib', 'poplib', 'posix', 'posixpath', 'pprint', 'profile',
    'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc',
    'queue', 'quopri', 'random', 're', 'readline', 'reprlib',
    'resource', 'rlcompleter', 'runpy', 'sched', 'secrets', 'select',
    'selectors', 'shelve', 'shlex', 'shutil', 'signal', 'site',
    'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd',
    'sqlite3', 'sre_compile', 'sre_constants', 'sre_parse', 'ssl',
    'stat', 'statistics', 'string', 'stringprep', 'struct', 'subprocess',
    'sunau', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny',
    'tarfile', 'telnetlib', 'tempfile', 'termios', 'test', 'textwrap',
    'threading', 'time', 'timeit', 'tkinter', 'token', 'tokenize',
    'tomllib', 'trace', 'traceback', 'tracemalloc', 'tty', 'turtle',
    'turtledemo', 'types', 'typing', 'unicodedata', 'unittest', 'urllib',
    'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser',
    'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc',
    'zipapp', 'zipfile', 'zipimport', 'zlib', '_thread',
}


def extract_top_level_module(import_name: str) -> str:
    """提取导入语句中的顶层包名。"""
    return import_name.split('.')[0]


def extract_imports_from_code(code: str) -> set:
    """
    从Python源码中提取所有导入的顶层模块名。
    先尝试AST解析，失败则退回正则匹配。
    """
    modules = set()

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(extract_top_level_module(alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.add(extract_top_level_module(node.module))
    except SyntaxError:
        # AST解析失败，退回正则
        for line in code.splitlines():
            line = line.strip()
            match = re.match(r'^import\s+([\w.]+)', line)
            if match:
                modules.add(extract_top_level_module(match.group(1)))
            match = re.match(r'^from\s+([\w.]+)\s+import', line)
            if match:
                modules.add(extract_top_level_module(match.group(1)))

    return modules


def filter_third_party(modules: set) -> set:
    """过滤掉标准库模块，保留第三方依赖。"""
    return {m for m in modules if m not in STDLIB_MODULES and not m.startswith('_')}


# 已知的模块名到PyPI包名的映射
MODULE_TO_PACKAGE = {
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'skimage': 'scikit-image',
    'osgeo': 'GDAL',
    'PIL': 'Pillow',
    'yaml': 'PyYAML',
    'bs4': 'beautifulsoup4',
    'attr': 'attrs',
    'dateutil': 'python-dateutil',
    'mpl_toolkits': 'matplotlib',
    'pysal': 'pysal',
    'mgwr': 'mgwr',
    'esda': 'esda',
    'libpysal': 'libpysal',
    'spreg': 'spreg',
    'pointpats': 'pointpats',
    'splot': 'splot',
    'mapclassify': 'mapclassify',
    'momepy': 'momepy',
    'networkx': 'networkx',
    'rasterstats': 'rasterstats',
    'richdem': 'richdem',
    'whitebox': 'whitebox',
    'xarray': 'xarray',
    'geocube': 'geocube',
    'exactextract': 'exactextract',
}


def module_to_package(module_name: str) -> str:
    """将模块名映射为pip可安装的包名。"""
    return MODULE_TO_PACKAGE.get(module_name, module_name)


def main():
    dataset_path = Path("dataset/GeoAnalystBench.csv")
    if not dataset_path.exists():
        print(f"数据集文件不存在：{dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)

    # 筛选开源任务
    opensource_df = df[df['Open Source'] == 'T']
    print(f"开源任务数量：{len(opensource_df)}")

    all_modules = set()
    per_task = {}

    for idx, row in opensource_df.iterrows():
        task_id = idx + 1
        code = row.get('CodeString', '')
        if not isinstance(code, str) or not code.strip():
            continue

        modules = extract_imports_from_code(code)
        third_party = filter_third_party(modules)
        per_task[task_id] = sorted(third_party)
        all_modules.update(third_party)

    # 汇总输出
    print(f"\n{'=' * 60}")
    print(f"第三方依赖汇总（{len(all_modules)}个模块）")
    print(f"{'=' * 60}\n")

    print("模块名 -> 包名")
    print("-" * 40)
    for module in sorted(all_modules):
        package = module_to_package(module)
        marker = " *" if module in MODULE_TO_PACKAGE else ""
        print(f"  {module:20s} -> {package}{marker}")

    print(f"\n{'=' * 60}")
    print("pip install 命令")
    print(f"{'=' * 60}\n")

    packages = sorted(set(module_to_package(m) for m in all_modules))
    print(f"pip install {' '.join(packages)}")

    # 逐任务明细
    print(f"\n{'=' * 60}")
    print("逐任务依赖明细")
    print(f"{'=' * 60}\n")

    for task_id in sorted(per_task.keys()):
        modules = per_task[task_id]
        if modules:
            print(f"  Task {task_id:2d}: {', '.join(modules)}")

    # 标记映射表中未覆盖的模块
    unmapped = {m for m in all_modules if m not in MODULE_TO_PACKAGE and m.replace('-', '_') != m}
    if unmapped:
        print(f"\n⚠ 以下模块的包名映射可能需要人工确认：{sorted(unmapped)}")


if __name__ == "__main__":
    main()