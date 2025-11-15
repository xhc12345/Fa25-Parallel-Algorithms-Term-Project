# Training and Running OpenCL Model

The folder contains two CNN models that are based on the same architecture with the same training process.

To ensure the OpenCL model runs, you must install `pytorch_dlprim` PyTorch backend from [GitHub](https://github.com/artyom-beilis/pytorch_dlprim/releases). Make sure you have the correct Python and PyTorch versions, along with drivers that support OpenCL. I recommend using the `requirements.txt` found in this folder to create a virtual environment.

After downloading the correct `.whl` backend, run `pip install <filename.whl>` to install it for you environment, after which in `CNN-MNIST-OCL.ipynb` select the kernel with this backend installed to train and run the model in OpenCL.

The author of `pytorch_dlprim` has stated that the training speed is about 60% of CUDA, inferencing is 80%.