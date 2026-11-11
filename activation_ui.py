"""Reusable, interactive activation-function visualizer.

Three distinct Streamlit entry points call run_app() with one fixed activation.
"""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from activation_core import activate, derivative, EXPLANATIONS


def run_app(name: str, configure: bool = True) -> None:
    info = EXPLANATIONS[name]
    if configure:
        st.set_page_config(page_title=f'{name} | BSD3513 Lab 4', page_icon='🧠', layout='wide')
    st.title(f'🧠 {name} Activation Function')
    st.caption('BSD3513 • Chapter 4: Neural Networks • Lab Report 4 • Student ID: SD24081')
    st.write('Explore the activation curve, gradient, and the output of a simulated artificial neuron.')
    with st.sidebar:
        st.header('Plot settings')
        x_limit = st.slider('Horizontal axis extent (±x)', 2, 20, 8, 1)
        points = st.slider('Curve resolution', 100, 1000, 400, 100)
        show_derivative = st.checkbox('Show derivative curve', True)
        st.divider()
        st.header('Neuron inputs')
        x1 = st.slider('Input x₁', -5.0, 5.0, 1.0, 0.1)
        x2 = st.slider('Input x₂', -5.0, 5.0, -0.5, 0.1)
        w1 = st.slider('Weight w₁', -4.0, 4.0, 1.5, 0.1)
        w2 = st.slider('Weight w₂', -4.0, 4.0, 0.8, 0.1)
        bias = st.slider('Bias b', -4.0, 4.0, 0.5, 0.1)
    z = w1*x1 + w2*x2 + bias
    y = float(activate(name, z))
    g = float(derivative(name, z))
    x = np.linspace(-float(x_limit), float(x_limit), points)
    ys, grads = activate(name, x), derivative(name, x)
    st.subheader('1. Activation function and gradient')
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4.3))
        ax.plot(x, ys, label=f'{name} f(x)', linewidth=2.5)
        if show_derivative:
            ax.plot(x, grads, label="Derivative f'(x)", linestyle='--', linewidth=2)
        ax.axhline(0, linewidth=0.8, color='gray')
        ax.axvline(0, linewidth=0.8, color='gray')
        if -x_limit <= z <= x_limit:
            ax.scatter([z], [y], s=100, marker='o', label=f'Neuron (z={z:.2f})', zorder=5)
        ax.set(xlabel='Weighted input z', ylabel='Function value / gradient', title=f'{name}: response to input')
        ax.grid(alpha=0.2)
        ax.legend(loc='best')
        st.pyplot(fig, use_container_width=True)
        output = io.BytesIO()
        fig.savefig(output, format='png', dpi=180, bbox_inches='tight')
        st.download_button('Download graph (PNG)', output.getvalue(), file_name=f'{name.lower()}_activation.png', mime='image/png')
        plt.close(fig)
    with col2:
        st.metric('Weighted sum z', f'{z:.4f}')
        st.metric('Activated output f(z)', f'{y:.4f}')
        st.metric("Gradient f'(z)", f'{g:.4f}')
        st.write('**Output range:**', info['range'])
        st.write('**Formula:**')
        st.latex(info['formula'])
        st.write('**Derivative:**')
        st.latex(info['derivative'])
    st.subheader('2. Forward pass through one neuron')
    st.latex(r'z = w_1x_1 + w_2x_2 + b')
    st.code(f'z = ({w1:.2f} × {x1:.2f}) + ({w2:.2f} × {x2:.2f}) + ({bias:.2f}) = {z:.4f}\nf(z) = {y:.4f}\nf\'(z) = {g:.4f}', language='text')
    st.subheader('3. Interpretation')
    st.write('**Common use:**', info['use'])
    st.write('**Strength:**', info['advantage'])
    st.write('**Limitation:**', info['limitation'])
    st.info(info['detail'])
    st.subheader('4. Download and verify numeric samples')
    table = pd.DataFrame({'Input x': x, f'{name} output': ys, 'Derivative': grads})
    st.dataframe(table.iloc[::max(1, len(table)//12)].round(5), hide_index=True, use_container_width=True)
    st.download_button('Download complete curve (CSV)', table.to_csv(index=False).encode(), file_name=f'{name.lower()}_curve.csv', mime='text/csv')
    st.caption('The curves show mathematical activation functions; this application does not train a full neural network.')
