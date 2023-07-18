Installations-Guide
====================================

Install
-------
If all external dependencies are available, a simple pip install alhazen-py suffices. We recommend installing alhazen-py inside a virtual environment (virtualenv):

..  code-block:: bash

    python3.10 -m venv venv
    source venv/bin/activate

    pip install --upgrade pip
    pip install alhazen-py

Now, the alhazen command should be available on the command line within the virtual environment.

Development and Testing
-----------------------
For development, we recommend using alhazen-py inside a virtual environment (virtualenv). By thing the following steps in a standard shell (bash), one can run the Alhazen tests:

..  code-block:: bash

    git clone https://github.com/martineberlein/alhazen-py.git
    cd alhazen-py/

    python3.10 -m venv venv
    source venv/bin/activate

    pip install --upgrade pip

    # Run tests
    pip install -e .[dev]
    python3 -m pytest

Build
-----
**alhazen-py** is build locally as follows:

..  code-block:: bash

    git clone https://github.com/martineberlein/alhazen-py.git
    cd alhazen-py/

    python3.10 -m venv venv
    source venv/bin/activate

    pip install --upgrade pip
    pip install --upgrade build
    python3 -m build

Then, you will find the built wheel (\*.whl) in the dist/ directory.
