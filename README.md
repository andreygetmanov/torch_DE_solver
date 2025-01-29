<p align="center">
    <img src="https://itmo.ru/file/pages/213/logo_na_plashke_russkiy_belyy.png" align="center" width="70%">
</p>
<p align="center"><h1 align="center">TORCH_DE_SOLVER</h1></p>
<p align="center">
	<img src="https://img.shields.io/github/license/andreygetmanov/torch_DE_solver?style=default&logo=opensourceinitiative&logoColor=white&color=blue" alt="license">
	<img src="https://img.shields.io/github/last-commit/andreygetmanov/torch_DE_solver?style=default&logo=git&logoColor=white&color=blue" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/andreygetmanov/torch_DE_solver?style=default&color=blue" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/andreygetmanov/torch_DE_solver?style=default&color=blue" alt="repo-language-count">
</p>
<p align="center">Built with the tools and technologies:</p>
<p align="center">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white"alt="NumPy">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white"alt="Python">
	<img src="https://img.shields.io/badge/GitHub%20Actions-2088FF.svg?style=default&logo=GitHub-Actions&logoColor=white"alt="GitHub%20Actions">
	<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=default&logo=SciPy&logoColor=white"alt="SciPy">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=default&logo=pandas&logoColor=white"alt="pandas">
</p>
<br>


---
## Overview

torchDEsolver is a Python package that helps solve partial differential equations (PDEs) using neural networks. It provides a flexible framework for defining custom loss functions, allowing users to train neural networks to approximate PDE solutions. By leveraging PyTorch, torchDEsolver enables efficient and scalable training of neural networks for a wide range of PDE problems. Its key features include support for various loss functions, including mean squared error with derivatives, and higher-order derivatives, making it a valuable tool for researchers and engineers working with PDEs.

---


## Table of contents

- [Core features](#core-features)
- [Installation](#installation)
- [Examples](#examples)
- [Documentation](#documentation)
- [Getting started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contacts](#contacts)

---

## Core features

1. **PyTorch Integration**: Utilizes `PyTorch` for building and training neural networks to solve partial differential equations.

2. **Customizable Loss Functions**: Provides a base `Loss` class and subclasses for different loss functions, including mean squared error (MSE) with and without derivatives.

3. **Finite Difference Calculations**: Employs `finitediffs` function to calculate derivatives and higher-order derivatives for loss function calculations.

4. **Modular Design**: Enables creation of custom loss functions for specific problems, allowing for flexibility in the training process.

5. **GPU Acceleration**: Supports GPU acceleration using `PyTorch` and `Device` class for efficient computations.

---


## Installation

Install torch_DE_solver using one of the following methods:

**Build from source:**

1. Clone the torch_DE_solver repository:
```sh
❯ git clone https://github.com/andreygetmanov/torch_DE_solver
```

2. Navigate to the project directory:
```sh
❯ cd torch_DE_solver
```

3. Install the project dependencies:


**Using `pip`** &nbsp;
[<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r requirements.txt
```



---


## Examples

Examples of how this should work and how it should be used are available in [examples](https://github.com/andreygetmanov/torch_DE_solver/tree/main/examples).

---


## Documentation

A detailed torch_DE_solver description is available in [docs](https://torch-de-solver.readthedocs.io).

---


## Getting started

### Usage

Run torch_DE_solver using the following command:
**Using `pip`** &nbsp;
[<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python {entrypoint}
```


---


## Contributing


- **[Report Issues](https://github.com/andreygetmanov/torch_DE_solver/issues )**: Submit bugs found or log feature requests for the torch_DE_solver project.


---


## License

This project is protected under the BSD 3-Clause "New" or "Revised" License. For more details, refer to the [LICENSE](https://github.com/andreygetmanov/torch_DE_solver/blob/main/LICENCE) file.

---


## Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---



## Contacts

Your contacts. For example:

- [Telegram channel](https://t.me/) answering questions about your project
- [VK group](<https://vk.com/>) your VK group
- etc.

---

