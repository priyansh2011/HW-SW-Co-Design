# Course Project - Hardware Software Co-Design

This repository contains the code for our course project:

The files are:

1. Design_Credit_Last_Sem.ipynb - This file contains the code for image classification task.

2. Bi_LSTM (1).ipynb, Fnet_Vs_Transformer_1.ipynb, Switch_transformer (1).ipynb - Contains the code for text classification task.

3. Model_{1,2,3}.py - Contains the code for DL on jet tagging data set, compiling and synthesising using hls4ml and their optimisations and their synthesis. 

## How to run codes..

For .ipynb you can open in Jupyter notebook or use google colab. Other option is to download them as .py  and then use.

For .py files  do:

```bash
python <file_name>.py
```

## How to run code using hls4ml

Follow these steps for smooth setup of the environment, the process is long and if  using the process given on the base repository we face a lot of errors.

Step1-In a linux based system do the following:
```bash
git clone https://github.com/hls-fpga-machine-learning/hls4ml-tutorial.git
cd hls4ml-tutorial
docker build --build-arg NB_USER=jovyan --build-arg NB_UID=1000 . -f 
```
Step2- Check if we got the image by doing 
```bash
docker images
```
Step3 - Get vivado from internet or the EEL_7210_2023@10.9.1.79 server.

Step4- Now, we have our image ready. We can create a container by doing the following:
```bash
docker run -it -t -v $PWD:/<directory in which Vivado is saved>:/workspace <image_id> bash 
```
Step5 - At this point we would be inside the container. Then add the vivado to the path. Assuming you downloaded 2019.2 version. Add this to the path:/Vivado/2019.2/bin. 

```bash
export PATH=$PATH:/Vivado/2019.2/bin:
```
Step6 - Then we have our environment ready, we can now use hls4ml as well as synthesise the models.

Step7 - Run the files from workspace directory. Make sure inside the code files, you save all the generated files from code in ../home/jovyan directory.

To run code do:
```bash
 python <code file name>.py
```

