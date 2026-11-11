import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Activation Function", layout="wide")
st.title("Rectified Linear Unit (ReLU) Activation Function")
st.write("ReLU returns zero for negative inputs and returns the input itself for positive inputs.")
st.latex(r"f(x) = \\max(0, x)")

st.sidebar.header("Input Controls")
x = st.sidebar.slider("Choose an input value (x)", -10.0, 10.0, 2.0, 0.1)
y = max(0.0, x)

c1, c2 = st.columns(2)
c1.metric("Input (x)", f"{x:.2f}")
c2.metric("ReLU Output", f"{y:.2f}")

st.subheader("ReLU Function Graph")
xgrid = np.linspace(-10, 10, 400)
ygrid = np.maximum(0, xgrid)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(xgrid, ygrid, label="ReLU: max(0, x)", linewidth=2)
ax.scatter([x], [y], s=80, label="Selected input/output")
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.set(xlabel="Input (x)", ylabel="Output f(x)", title="ReLU Activation Function")
ax.grid(True)
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.subheader("Interpretation")
if x < 0:
    st.write(f"The input is negative ({x:.2f}), so ReLU returns 0.")
elif x > 0:
    st.write(f"The input is positive ({x:.2f}), so ReLU returns the same value ({y:.2f}).")
else:
    st.write("When the input is 0, the ReLU output is also 0.")
st.info("ReLU introduces non-linearity into a neural network and is simple to compute.")
