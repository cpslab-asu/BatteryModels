# Code for Pys-TaLiRo


# Installation

We use the [Poetry tool](https://python-poetry.org/docs/) which is a dependency management and packaging tool in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Please follow the installation of poetry at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)

After you've installed poetry, you can install partx by running the following command in the root of the project: 

```
poetry install
```

# Models

1. Matlab Model: ``test_fan_aircooling_3.slx``
2. Python Model: ``pythonBatteryModel.py``

Run ``test_run.py`` to conduct an initial state.


# Versions:

Required MathWorks Products
-    MATLAB release R2023b
-    Simulink
-    SimScape Electrical
-    SimScape Fluids

Required 3rd Party Products

-   Python 
-   For matlab and python integration, please refer to [https://pypi.org/project/matlabengine/23.2.3/](https://pypi.org/project/matlabengine/23.2.3/)