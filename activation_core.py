"""Pure numerical functions shared by the three Streamlit apps.

BSD3513 Introduction to AI | Lab 4 | SD24081
No Streamlit dependencies in this module: can be unit tested independently.
"""
from __future__ import annotations
import numpy as np

FUNCTIONS = ('ReLU', 'Sigmoid', 'Tanh')


def activate(name: str, x):
    """Apply a numerically stable activation to an array or scalar."""
    a = np.asarray(x, dtype=float)
    if name == 'ReLU':
        return np.maximum(a, 0.0)
    if name == 'Sigmoid':
        # Stable expit avoids overflow for very negative/positive inputs.
        return np.where(a >= 0, 1.0 / (1.0 + np.exp(-np.maximum(a, 0))),
                        np.exp(np.minimum(a, 0)) / (1.0 + np.exp(np.minimum(a, 0))))
    if name == 'Tanh':
        return np.tanh(a)
    raise ValueError(f'Unknown activation function: {name}')


def derivative(name: str, x):
    """Standard derivatives; ReLU uses the conventional subgradient 0 at x=0."""
    a = np.asarray(x, dtype=float)
    if name == 'ReLU':
        return (a > 0).astype(float)
    y = activate(name, a)
    if name == 'Sigmoid':
        return y * (1.0 - y)
    if name == 'Tanh':
        return 1.0 - y*y
    raise ValueError(f'Unknown activation function: {name}')


EXPLANATIONS = {
    'ReLU': {
        'formula': r'f(x)=\max(0,x)',
        'derivative': r"f'(x)=\begin{cases}0 & x\leq 0\\1 & x>0\end{cases}",
        'range': '[0, infinity)',
        'use': 'Often used in hidden layers; simple and computationally cheap.',
        'advantage': 'Retains gradients for positive inputs and is efficient.',
        'limitation': 'Zero gradient for negative inputs (dying ReLU); not differentiable at zero.',
        'detail': 'At x=0, the application defines the derivative/subgradient as 0.',
    },
    'Sigmoid': {
        'formula': r'f(x)=\frac{1}{1+e^{-x}}',
        'derivative': r"f'(x)=f(x)(1-f(x))",
        'range': '(0, 1)',
        'use': 'Often used for a binary-classification output probability.',
        'advantage': 'Outputs a smooth bounded value between 0 and 1.',
        'limitation': 'Gradient becomes very small for large positive or negative inputs (saturation).',
        'detail': 'At x=0, the sigmoid output is 0.5 and derivative is 0.25.',
    },
    'Tanh': {
        'formula': r'f(x)=\tanh(x)',
        'derivative': r"f'(x)=1-\tanh^2(x)",
        'range': '(-1, 1)',
        'use': 'Useful in hidden layers and some recurrent-network formulations.',
        'advantage': 'Outputs are zero-centered; derivative at zero is 1.',
        'limitation': 'Gradient saturates near -1 and 1 for large-magnitude inputs.',
        'detail': 'At x=0, tanh output is 0 and derivative is 1.',
    },
}
