## 🔍 Overview

The core design remains faithful to Karpathy’s minimalist implementation — 
a pure Python autodiff engine that supports backpropagation over scalar-valued computational graphs. 
This fork, however, **extends the engine with advanced neural network activation functions**, such as:

- ✅ `Softplus` (smooth ReLU alternative)
- ✅ `Swish` (sigmoid-weighted input)
- ✅ `LeakyReLU`, `ELU`, and more *(to be added incrementally)*

These additions make `micrograd_plus` suitable for **research-grade introspection**, 
**experimental optimization**, and **educational demonstrations** of gradient flow under various nonlinearities.

## 📁 Directory Structure

```text

├── micrograd_plus/
│   ├── __init__.py       # Package initializer
│   ├── engine.py         # Core autodiff engine (extended from micrograd)
│   ├── nn.py             # MLP framework with support for custom activations

📜 License

MIT — feel free to use and extend.

email:
leburikplc@gmail.com
