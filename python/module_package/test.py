# 1. no identical name .py(module) and folder(package) in the same dir

# ------------------------------------------------------------------------

# 2. for x.y.z, all of x and y must be packages

# ------------------------------------------------------------------------

# 3. from foo import func = import foo.func
## former use func to call, while latter use foo.func to call

# ------------------------------------------------------------------------

# 4. from x.y.z import k

## 4.1 when z is module(.py), k is a class/function
#from foo.bar.func import hello
#hello()

## 4.2 when z is a package(folder), k is a class/function of __init__.py or a module(.py)
#from foo.bar import func
#func.hello()

# ------------------------------------------------------------------------

# 5. import x always import an .py file

## 5.1 for module: x.py
#import foo.bar.func
#foo.bar.func.hello()

## 5.2 for package: x/__init__.py
#import foo.bar
#foo.bar.func.hello() # invalid because only import foo/bar/__init__.py

## 5.3 import will execute .py
#import foobar

# ------------------------------------------------------------------------

# 6. from .func import hello

## 6.1 cannot directly execute a module(.py) having relative import
#python baz/directly_execute.py

## 6.2 file executed cannot be in the same directory of intermedia file
#python baz/same_dir.py

## 6.3 shoule be non-directly && different directory
#from baz import hello
#hello()

## 6.4 explanation:
## relative path is based on __name__ of module
## when directly execute, directly executed's __name__ = __main__
## when in the same directory of intermedia file, intermedia's __name__ = intermedia, rather than pkg.intermedia
