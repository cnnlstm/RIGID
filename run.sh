mkdir -p log/main_full_batch_fast_reg_wo_noise_wo_reg
now=$(date +"%Y%m%d_%H%M%S")

# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python get_alignment.py 2>&1|tee log/train-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python get_latent.py 2>&1|tee log/train-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python get_framemask.py 2>&1|tee log/train-$now.log & 

# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:2 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python main_full_batch_fast_reg_wo_noise_train.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/train-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 python main_full.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/train-$now.log & 
srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python main_full.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/train-$now.log & 


# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:2 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 python main_full_batch_fast_reg_wo_noise_wo_st.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise_wo_st/train-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:2 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 python main_full_batch_fast_reg_wo_noise_current_input.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise_current_input/train-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:2 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 python main_full_batch_fast_reg_wo_noise_wo_reg.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise_wo_reg/train-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:4 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 python main_full_batch_fast_reg_wo_noise_wo_reg.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise_wo_reg/train-$now.log & 

# srun --partition Gveval3 --mpi=pmi2 --gres=gpu:2 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 python main_full_batch_fast_reg_wo_noise_wo_edit.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise_wo_edit/train-$now.log & 


# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1  python main_full_batch_fast_reg_wo_noise_test_edit.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 

# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1  --quotatype=auto  python   main_full_batch_fast_reg_wo_noise_test_anime.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 




# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python main_full_batch_fast_reg_wo_noise_current_input_test_edit.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python main_full_batch_fast_reg_wo_noise_current_input_test_inversion.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 

# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python main_full_batch_fast_reg_wo_noise_different_layer_test_edit.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python main_full_batch_fast_reg_wo_noise_different_layer_test_inversion.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 

# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python main_full_batch_fast_reg_wo_noise_test_edit.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python main_full_batch_fast_reg_wo_noise_test_inversion.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 


# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1  python main_full_batch_fast_reg_wo_noise_wo_edit_test_edit.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 python main_full_batch_fast_reg_wo_noise_wo_edit_test_inversion.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 

# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1  python main_full_batch_fast_reg_wo_noise_wo_reg_test_edit.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1  python main_full_batch_fast_reg_wo_noise_wo_reg_test_inversion.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 

# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1  python main_full_batch_fast_reg_wo_noise_wo_st_test_edit.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1  python main_full_batch_fast_reg_wo_noise_wo_st_test_inversion.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 




# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python main_full_batch_fast_reg_wo_noise_wo_st_test_edit.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python main_full_batch_fast_reg_wo_noise_wo_edit_test_edit.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise_wo_edit/test-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python main_full_batch_fast_reg_wo_noise_different_layer_test_edit.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 


# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python main_full_batch_fast_reg_wo_noise_test_inversion.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python main_full_batch_fast_reg_wo_noise_current_input_test_inversion.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 


# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python evaluation/lpip.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 
# srun --partition Gveval3 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python compute_flow_occlusion.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python evaluation/warp_error.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python evaluation/id.py 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python evaluation/tool/metrics/metric_center.py --root_dir / --path_gen /mnt/petrelfs/xuyangyang.p/RIGID/comparison/edit/IGCI --path_gt /mnt/petrelfs/xuyangyang.p/RIGID/comparison/ablation/GT/Inversion_Unaligned --type fvd --write_metric_to ./evaluation/RIGID-Highf-2_FVD_16frames.txt --number_sample_frames 16 --sample_duration 16 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 
# srun --partition Gveval2 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=yangyangxu --kill-on-bad-exit=1 --quotatype=auto python evaluation/tool/metrics/metric_center.py --root_dir / --path_gen ./evaluation/CELEBV-HQ-Unseen-Test/recontruction_wo_edit/ --path_gt ./evaluation/CELEBV-HQ-Unseen-Test/original/ --type fvd --write_metric_to ./evaluation/CELEBV-HQ-Unseen-Test/FULL_Batch_FAST_REG_wo_Noise_wo_Edit_FVD_16frames.txt --number_sample_frames 16 --sample_duration 16 2>&1|tee log/main_full_batch_fast_reg_wo_noise/test-$now.log & 


