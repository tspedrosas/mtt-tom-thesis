# LLM ToM Machine Teaching

Machine teaching is the application of machine learning techniques to transfer knowledge between AI models. In this project
we focus on using Theory of Mind to guide the knowledge transfer between LLMs.

Using Theory of Mind to model a student model allows for the teacher to have a more correct assessment of the learning
process. So, we combine ToM-like approaches to have teacher LLMs model a student model in order to decide when and how 
to intervene.

## Installation

We provide three installation scripts:

- *install_miniconda.sh* - is used to install miniconda if you don't have it installed

  > source install_miniconda.sh [\<path-install-miniconda\>] 

- *install_python_jax_env.sh* - used to install a python virtual environment (either using conda or venv+pip) that uses JAX for CUDA support

  > source \<path-to-dir\>/install_python_jax_env.sh -n \<env-name\> -t \<conda|pip\> [-p \<python-version\>] [-c \<path-conda-home\>]

- *install_python_pytorch_env.sh* - used to install a python virtual environment (either using conda or venv+pip) that uses Pytorch for CUDA support
  
  > source \<path-to-dir\>/install_python_pytorch_env.sh -n \<env-name\> -t \<conda|pip\> [-p \<python-version\>] [-c \<path-conda-home\>]
 
## Run

In the *scripts* folder there are scripts for testing different ToM-like approaches -- static student model from 
Saha et al.[1], evolving student model, etc... -- to model student models and guide the teaching processes. These scripts
also support slurm usage and vllm model serving, just need to adapt the sbatch parameters for your cluster configurations.


### References

[1] - Saha, S., Hase, P., & Bansal, M. (2023). Can language models teach weaker agents? teacher explanations improve students via theory of mind. arXiv preprint arXiv:2306.09299.