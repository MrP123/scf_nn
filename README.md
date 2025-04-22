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

## Usage
Each Jupyter notebook represents one kind of model that is being trained. The shared dataset generation,, model definition, training loop and testing is implemented in the respective modules `nn_helper.py`, `nn_models.py`, `nn_train.py`.

## Literature references
- [1] J. Yu, J. Liu, Z. Peng, L. Gan, and S. Wan, “Localization of impact on CFRP structure based on fiber Bragg gratings and CNN-LSTM-Attention,” Optical Fiber Technology, vol. 87, p. 103943, Oct. 2024, doi: 10.1016/j.yofte.2024.103943.
- [2] K.-C. Jung and S.-H. Chang, “Advanced deep learning model-based impact characterization method for composite laminates,” Composites Science and Technology, vol. 207, p. 108713, May 2021, doi: 10.1016/j.compscitech.2021.108713.
- [3] K.-C. Jung, M.-G. Han, and S.-H. Chang, “Impact characterisation of draped composite structures made of plain-weave carbon/epoxy prepregs utilising smart grid fabric consisting of ferroelectric ribbon sensors,” Composite Structures, vol. 238, p. 111940, Apr. 2020, doi: 10.1016/j.compstruct.2020.111940.
