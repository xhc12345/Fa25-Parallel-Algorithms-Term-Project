# CNN Demo App

This folder contains a simple Flask web backend server that hosts a simple local website that provides an interface for you to interact with the CNN models.

You must ensure you are using Python 3.12.

Install the necessary dependencies found in `requirements.txt` in this folder using the command:

```ps1
pip install -r requirements.txt
```

You might want to create a virtual environment for this web app before running the above command as good engineering practice. Google how to create one if you are interested.

## Starting the web app

Once you have the pip requirements installed, you must ensure you have the PyTorch CNN models built for YOUR hardwares before you can start the server.

Please navigate to `./models` folder for further instruction.

Once you have FP32 CPU model and INT8 CPU model built, you can start this appliation by running this command from this folder:

```ps1
.\run.ps1
```