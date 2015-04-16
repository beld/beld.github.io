---
layout: post
title: Python classes
---

  Python’s class mechanism adds classes with a minimum of new syntax and semantics. It is a mixture of the class mechanisms found in C++ and Modula-3. Python classes provide all the standard features of Object Oriented Programming: the class inheritance mechanism allows multiple base classes, a derived class can override any methods of its base class or classes, and a method can call the method of a base class with the same name. Objects can contain arbitrary amounts and kinds of data. As is true for modules, classes partake of the dynamic nature of Python: they are created at runtime, and can be modified further after creation.
  
  Aliases behave like pointers in some respects. For example, passing an object is cheap since only a pointer is passed by the implementation; and if a function modifies an object passed as an argument, the caller will see the change.
  
  A namespace is a mapping from names to objects. Most namespaces are currently implemented as Python dictionaries. Examples of namespaces are: 
  the set of built-in names (containing functions such as abs(), and built-in exception names); 
  the global names in a module;
  the local names in a function invocation. 
  In a sense the set of attributes of an object also form a namespace. 
The important thing to know about namespaces is that there is absolutely no relation between names in different namespaces; for instance, two different modules may both define a function maximize without confusion — users of the modules must prefix it with the module name.
  
  Namespaces are created at different moments and have different lifetimes.
1. The namespace containing the built-in names is created when the Python interpreter starts up, and is never deleted.
2. The global namespace for a module is created when the module definition is read in; normally, module namespaces also last until the interpreter quits.
3. The statements executed by the top-level invocation of the interpreter, either read from a script file or interactively, are considered part of a module called __main__, so they have their own global namespace. (The built-in names actually also live in a module; this is called __builtin__.)
4. The local namespace for a function is created when the function is called, and deleted when the function returns or raises an exception that is not handled within the function. 

  A scope is a textual region of a Python program where a namespace is directly accessible. “Directly accessible” here means that an unqualified reference to a name attempts to find the name in the namespace.
  At any time during execution, there are at least three nested scopes whose namespaces are directly accessible:
1. the innermost scope, which is searched first, contains the local names
2. the scopes of any enclosing functions, which are searched starting with the nearest enclosing scope, contains      non-local, but also non-global names
3. the next-to-last scope contains the current module’s global names
4. the outermost scope (searched last) is the namespace containing built-in names

Class objects support two kinds of operations: attribute references and instantiation. 

class MyClass:
    """A simple example class"""
    i = 12345
    def f(self):
        return 'hello world'
        
Attribute references use the standard syntax used for all attribute references in Python: obj.name. MyClass.i and MyClass.f are valid attribute references.
Class instantiation uses function notation. x = MyClass() reates a new instance of the class and assigns this object to the local variable x.

The instantiation operation (“calling” a class object) creates an empty object.
def __init__(self):
    self.data = []
When a class defines an __init__() method, class instantiation automatically invokes __init__() for the newly-created class instance.

The only operations understood by instance objects are attribute references. There are two kinds of valid attribute names, data attributes and methods.

Data attributes need not be declared; like local variables, they spring into existence when they are first assigned to.****
A method is a function that “belongs to” an object. In Python, the term method is not unique to class instances: other object types can have methods as well. For example, list objects have methods called append, insert, remove, sort, and so on.


