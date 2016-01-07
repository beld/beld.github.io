---
layout:     post
title:      "Object Oriented Concepts Refreshing"
subtitle:   ""
date:       2016-01-07 23:30:00
author:     "Beld"
header-img: "img/post-bg-engineer.jpg"
tags:
    - software engineering
---

#### OOP Foundations

##### From Phenomena and Concepts to Abstraction and Modeling

Phenomenon: An object in the world of a domain as you perceive <br />
Concept: Describes the common properties of phenomena, 3-tuple: name, purpose and members <br />
Abstraction: Classification of phenomena into concepts <br />
Modeling: Development of abstractions to answer specific questions about a set of phenomena while ignoring irrelevant details <br />

Type: A concept in the context of programming languages <br />
Instance: A member of a specific type <br />
Classes:
- A complex type is represented by a *class* <br />
- A class is a code template for a concept, that is used to create *instances* of that concept <br />

Objects: An *instance of a class* at runtime is called Object <br />

##### Four Features of Object-oriented Programming Languages
- Abstraction
- Inheritance
- Encapsulation
- Polymorphism

Abstraction: Creating a model of the problem in terms of classes and the relationships between them <br />
Encapsulation: Objects are self-contained set of data and behavior; An object can determine which part of its data and behavior is exposed to the outer world <br />

#### Coupling and Cohesion
- Coupling: Measures the dependencies between subsystems
- Cohesion: Measures the dependencies among classes within a subsystem
- Low coupling: <br />
•  The subsystems should be as independent of each other as possible <br />
•  A change in one subsystem should not affect any other subsystem
- High cohesion: <br />
•  A subsystem should only contain classes which depend heavily on each other.

What makes good design?
-  Good design reduces complexity
-  Good design is prepared for change
-  Good design provides low coupling
-  Good design provides high cohesion.

Law of Demeter (LoD)
A method M of an object O may only invoke the methods of the following kinds of objects:
1.  O itself
2.  M's parameters
3.  any objects created/instantiated within M 4.  O's direct component objects
4. For Java, this can be formulated as: “Use only one dot”: Use only 1 level of method calls

#### Polymorphism
General definition  
- The ability of an object to assume different forms or shapes  

Computer Science
- The ability of an abstraction to be realized in multiple ways
- The ability of an interface to be realized in multiple ways
- The dynamic treatment of objects based on their type.

Parametric Polymorphism  
Example: generic type List  
A type is called a generic type if it has a type parameter: ArrayList<E>  
A type parameter is a placeholder for a specific type  
Operations on generic types are called generic operations: add() and get()  

Inheritance  
Subtyping: According to Liskov, a subtype must satisfy the substitution principle  
Subclassing: Subclassing simply denotes the usage of an inheritance association  
Overriding: Overriding is a special case in subclassing where a method implemented in the superclass is reimplemented in the subclass; The selection of which method to execute at runtime allows to change the behavior of an object without extensive case distinctions.   

Ad-Hoc Polymorphism  
Overloading: The ability to let a feature name denote two or more operations  
Signature: Parameter types and return result type of a method  
The signature is used to decide which of the possible operations is meant by a particular instance of that name  
The compiler makes the decision.  

Binding  
－ Binding: establishes the mappings between names and data objects and their descriptions
－ Early Binding (Static binding, at compile time): The premature choice of operation variant, resulting in possibly wrong results and (in favorable cases) run-time system crashes   
－ Late Binding (Dynamic binding, at run time): The guarantee that every execution of an operation will select the correct version of the operation, based on the type of the operation’s target

Delegation  
A mechanism for code reuse in which an operation resends a message to another class to accomplish the desired behavior.   
Delegation simply involves passing a method call to another object, transforming the input if necessary   
Delegation extends the behavior of an object  
The Receiving Object delegates a request to perform an operation to its Delegate  
