# Impact Localization with Neural Networks in Smart Composite Fabrics (SCFs)
This project contains neural networks developed to localize low velocity impacts on smart composite fabrics (SCFs).
Those SCFs have embedded strain sensors based on piezoelectric PVDF foil.

Development and tests were carried out in the System Design with Advanced Composites Lab at Chung-Ang-University, Seoul.

Current implementations of the NN architectures are strongly based on literature making use of FBG based sensors in normal composites.

This project uses Pytorch. It requires `Python 3.12` and a CUDA capable GPU.

## Installation
1. Create a `venv` with `Python 3.12`
2. Install `graphviz` for plotting of the model architecture
3. Install `requirements.txt`