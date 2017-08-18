import pkg_resources

#ep = pkg_resources.EntryPoint.parse('x=pkg:Huyh')
ep = pkg_resources.EntryPoint('x', 'pkg', ('Huyh',))
print ep
instance = ep.load(False)
x = instance()

# advantage over import: no need to modify caller(add import xxx) when adding more callee
#
# given a new callee, e.g. foo.bar.func
#
# import style: 
# import foo.bar.func
# if(callee = 'foo.bar.func')
#   return foo.bar.func
#
# enrtypoint style: no need to modify
# EntryPoint('foo.bar.func')
