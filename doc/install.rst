.. _install:

Installing
==========

.. note::

    **This package is in early stages of development.**

    We welcome any feedback and ideas!
    Let us know by submitting
    `issues on Github <https://github.com/fatiando/deeplook/issues>`__
    or send us a message on our
    `Gitter chatroom <https://gitter.im/fatiando/deeplook>`__.


Which Python?
-------------

You'll need **Python 2.7, 3.5 or greater**.

We recommend using the `Anaconda <http://continuum.io/downloads#all>`__ Python
distribution to ensure you have all dependencies installed and the ``conda``
package manager available.
Installing Anaconda does not require administrative rights to your computer and
doesn't interfere with any other Python installations in your system.


Dependencies
------------

Deeplook requires the following libraries:

* numpy


Installing
----------

Use ``pip`` to install the latest source from Github::

    pip install https://github.com/fatiando/deeplook/archive/master.zip

Alternatively, you can clone the git repository and install using ``pip``::

    git clone https://github.com/fatiando/deeplook.git
    cd deeplook
    pip install .


Testing your install
--------------------

Deeplook ships with a full test suite.
You can run our tests after you install it but you will need a few extra
dependencies as well::

    conda install pytest -c conda-forge

Test your installation by running the following inside a Python interpreter::

    import deeplook
    deeplook.test()
