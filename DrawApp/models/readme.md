# PyTorch CNN Models for MNIST Digit Classifier

There are 3 distinct models you can build via the 2 provided Jupyter notebooks: `CNN-MNIST-CPU.ipynb` and `CNN-MNIST-OCL.ipynb`, where the former builds the CPU-only FP32 and INT8 models, and the latter builds the OpenCL-based GPU FP32 model.

## Training and Running CPU Model

1. Open `CNN-MNIST-CPU.ipynb` (I recommend VS Code with Jupyter extension)

2. Make sure no import statements give you errors (if any, run `pip install -r requirements.txt` in this folder)

3. Click run all (or equivalent)

4. Profit! (The two models automatically save to `./model_cpu/bin` folder)

### Running the CPU model

`Model-Runner-CPU.ipynb` provides minimum code needed to run and compare these two built models.

## Training and Running OpenCL Model

To ensure the OpenCL model runs, you must install `pytorch_dlprim` PyTorch backend from [GitHub](https://github.com/artyom-beilis/pytorch_dlprim/releases). Make sure you have the correct Python and PyTorch versions, along with drivers that support OpenCL. I recommend using the `requirements.txt` found in this folder to create a virtual environment.

After downloading the correct `.whl` backend, run:
```ps1
pip install <filename.whl>
```
Where `<filename.whl>` is the file name of the whl file you downloaded. This is the OpenCL backend. This install it for you environment, after which in `CNN-MNIST-OCL.ipynb` select the kernel that has this backend installed to train and run the model in OpenCL.

The author of `pytorch_dlprim` has stated that the training speed is about 60% of CUDA, inferencing is 80%.

### Running the OpenCL model

`Model-Runner-OCL.ipynb` provides minimum code needed to run and compare these two built models.