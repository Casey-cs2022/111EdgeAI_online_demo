''' w/o tvm '''
# run dmd model use video
python3 ./tvm_tsm/run_wotvm.py --model ./tvm_tsm/weights/dmd_finetune_TSM_e1000_b4_n4_mobilenetv2_lr0.000100_best.pth.tar --dataset dmd --video ./tvm_tsm/real_time_infer/IMG_1691.MOV --setting video

# run dmd model use camera
python3 ./tvm_tsm/run_wotvm.py --model ./tvm_tsm/weights/dmd_finetune_TSM_e1000_b4_n4_mobilenetv2_lr0.000100_best.pth.tar --dataset dmd --setting camera

