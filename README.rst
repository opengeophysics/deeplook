Deeplook
========

**A Python framework for solving inverse problems.**

`Documentation <https://opengeophysics.github.io/deeplook>`_ |
`Install <https://opengeophysics.github.io/deeplook/install.html>`_ |
`API <https://opengeophysics.github.io/deeplook/api>`_
`Contact <https://gitter.im/opengeophysics>`_

.. image:: http://img.shields.io/pypi/v/deeplook.svg?style=flat-square
    :alt: Latest version on PyPI
    :target: https://pypi.python.org/pypi/deeplook
.. image:: http://img.shields.io/travis/opengeophysics/deeplook/master.svg?style=flat-square&label=tests
    :alt: Travis CI build status
    :target: https://travis-ci.org/opengeophysics/deeplook
.. image:: https://img.shields.io/codecov/c/github/opengeophysics/deeplook/master.svg?style=flat-square
    :alt: Test coverage status
    :target: https://codecov.io/gh/opengeophysics/deeplook
.. image:: https://img.shields.io/codeclimate/maintainability/opengeophysics/deeplook.svg?style=flat-square
    :alt: Code quality status
    :target: https://codeclimate.com/github/opengeophysics/deeplook
.. image:: https://img.shields.io/codacy/grade/e73169dcb8454b3bb0f6cc5389b228b4.svg?style=flat-square&label=codacy
    :alt: Code quality grade on codacy
    :target: https://www.codacy.com/app/leouieda/deeplook
.. image:: https://img.shields.io/gitter/room/opengeophysics.svg?style=flat-square
    :alt: Chat room on Gitter
    :target: https://gitter.im/opengeophysics


Disclaimer
----------

**This package in early stages of design and implementation.**

We are at the stage of defining the design and goals of the project.
We welcome any ideas and participation from the community!
Let us know by submitting
`issues on Github <https://github.com/opengeophysics/deeplook/issues>`__
or send us a message on our
`Gitter chatroom <https://gitter.im/opengeophysics>`__.


Project goals
-------------

* Provide APIs on two levels: a high-level scikit-learn like API for users of
  inversions and lower-level API for developers of inversions.
* Python 3 from the start.
* Agnostic of the forward operator calculation.
* Provide tools to automate as much as possible (finite-difference derivatives,
  goal function normalization, etc).
* Flexible low-level API that allows complete customization of the inversion
  process.


Contacting Us
-------------

* Most discussion happens `on Github <https://github.com/opengeophysics/deeplook>`__.
  Feel free to `open an issue
  <https://github.com/opengeophysics/deeplook/issues/new>`__ or comment
  on any open issue or pull request.
* We have `chat room on Gitter <https://gitter.im/opengeophysics/>`__
  where you can ask questions and leave comments.


Contributing
------------

Code of conduct
+++++++++++++++

Please note that this project is released with a
`Contributor Code of Conduct <https://github.com/opengeophysics/deeplook/blob/master/CODE_OF_CONDUCT.md>`__.
By participating in this project you agree to abide by its terms.

Contributing Guidelines
+++++++++++++++++++++++

Please read our
`Contributing Guide <https://github.com/opengeophysics/deeplook/blob/master/CONTRIBUTING.md>`__
to see how you can help and give feedback.

Imposter syndrome disclaimer
++++++++++++++++++++++++++++

**We want your help.** No, really.

There may be a little voice inside your head that is telling you that you're
not ready to be an open source contributor; that your skills aren't nearly good
enough to contribute.
What could you possibly offer?

We assure you that the little voice in your head is wrong.

**Being a contributor doesn't just mean writing code**.
Equality important contributions include:
writing or proof-reading documentation, suggesting or implementing tests, or
even giving feedback about the project (including giving feedback about the
contribution process).
If you're coming to the project with fresh eyes, you might see the errors and
assumptions that seasoned contributors have glossed over.
If you can write any code at all, you can contribute code to open source.
We are constantly trying out new skills, making mistakes, and learning from
those mistakes.
That's how we all improve and we are happy to help others learn.

*This disclaimer was adapted from the*
`MetPy project <https://github.com/Unidata/MetPy>`__.


Related projects
----------------

* [Fatiando a Terra](http://www.fatiando.org)
* [SimPEG](http://simpeg.xyz/)
* [pyGMILi](https://www.pygimli.org/)


License
-------

This is free software: you can redistribute it and/or modify it under the terms
of the **BSD 3-clause License**. A copy of this license is provided in
``LICENSE.txt``.
