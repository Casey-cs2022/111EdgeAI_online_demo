# 111EdgeAI_online_demo
# Implementation on Jetson Nano
The code is based on tsm github and tvm official website. 
TSM : https://github.com/mit-han-lab/temporal-shift-module,
TVM : https://tvm.apache.org/docs/tutorial/autotvm_relay_x86.html


## Environment setup
Please follow the step to install the environment.

```
conda create -n video python=3.6
conda activate video
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```


## Repository structure
```
online_demo
├── tvm_non_local
    ├── weights                 # The network weights folder
    ├── mobilenet_v2_tsm_nl.py  # The network architecture of tsm with non local operation 
    ├── non_local.py            # The code for non local operation
    ├── run_wotvm.py            # Run the online demo without tvm
├── tvm_tsm
    ├── real_time_infer         # The real-time video
    ├── weights                 # The network weights folder
    ├── mobilenet_v2_tsm.py     # The network architecture of tsm
    ├── run_wotvm.py            # Run the online demo without tvm
```


## Run the online demo
The two folders **tvm_non_local** and **tvm_tsm** represent run the online demo using the model with non-local module and run the online demo using the model without non-local module.

When we run without tvm compiler, the command is : 

```
python3 run_wotvm.py \
--model ./weights/dmd_finetune_TSM_e1000_b4_n4_mobilenetv2_lr0.000100_best.pth.tar \
--dataset dmd --video /home/anomaly/Desktop/video/dmd_1.mp4 \
--setting video
```

```
python3 run_wotvm.py \
--model ./weights/dmd_finetune_TSM_e1000_b4_n4_mobilenetv2_lr0.000100_best.pth.tar \
--dataset dmd --setting camera
```


## Argument for online demo
```
model : the network weights we want to used.
mode : the mode in tvm we want to used. e.g. tvm, autotvm, autoscheduler (if we run without tvm, we don't need this)
dataset : to determine the same classes with the dataset.
video : if the 'setting' is video, we use this to determine which video we want to test.
setting : The setting is mean the loading mode. e.g. camera, video
```

