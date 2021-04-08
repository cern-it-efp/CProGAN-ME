# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

#----------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".
import numpy as np
import tensorflow as tf
import os
class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

#----------------------------------------------------------------------------
# Paths.

data_dir ='/root/CProGAN-ME/datasets/person'
result_dir ='results'

#----------------------------------------------------------------------------
# TensorFlow options.

tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()        # Environment variables, set by the main program in train.py.

tf_config['graph_options.place_pruned_graph']   = True      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
#tf_config['gpu_options.allow_growth']          = False     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
env.TF_CPP_MIN_LOG_LEVEL                        = '1'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.

#----------------------------------------------------------------------------
# Official training configs, targeted mainly for CelebA-HQ.
# To run, comment/uncomment the lines as appropriate and launch train.py.

desc        = 'pgan'                                        # Description string included in result subdir name.
random_seed = 1000                                          # Global random seed.
dataset     = EasyDict()                                    # Options for dataset.load_dataset().
train       = EasyDict(func='train.train_progressive_gan')  # Options for main training func.
G           = EasyDict(func='networks.G_paper')             # Options for generator network.
D           = EasyDict(func='networks.D_paper')
G_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for generator optimizer.
D_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)
G_loss      = EasyDict(func='loss.G_wgan_acgan')            # Options for generator loss.
D_loss      = EasyDict(func='loss.D_wgangp_acgan')          # Options for discriminator loss.
sched       = EasyDict(lod_initial_resolution = 32)          # Options for train.TrainingSchedule.
grid        = EasyDict(size='1080p', layout='random')       # Options for train.setup_snapshot_image_grid().

# Dataset
desc += '-syn256rgb_conditional';          dataset = EasyDict(tfrecord_dir= data_dir)

# Config presets
bmark_gpus = int(os.environ['BMARK_GPUS'])
bmark_kimg = int(os.environ['BMARK_KIMG'])

if bmark_gpus == 1:
    desc += '-preset-v2-1gpu'; num_gpus = 1; sched.minibatch_base = 4; sched.minibatch_dict = {4: 64, 8: 64, 16: 64, 32: 32, 64: 16, 128: 8, 256: 4, 512: 2}; sched.G_lrate_dict = {1024: 0.0015}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = bmark_kimg # 12000
elif bmark_gpus == 2:
    desc += '-preset-v2-2gpus'; num_gpus = 2; D.mbstd_group_size = 16; sched.minibatch_base = 16; sched.minibatch_dict = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}; sched.G_lrate_dict = {512: 0.0015, 1024: 0.002}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict);sched.lod_training_kimg = 300; sched.lod_transition_kimg = 1500; train.total_kimg = bmark_kimg # 10000
elif bmark_gpus == 4:
    desc += '-preset-v2-4gpus'; num_gpus = 4; sched.minibatch_base = 16; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}; sched.G_lrate_dict = {256: 0.0015, 512: 0.002, 1024: 0.003}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); sched.lod_training_kimg = 300; sched.lod_transition_kimg = 1500; train.total_kimg = bmark_kimg # 12000
else:
    desc += '-preset-v2-8gpus'; num_gpus = 8; sched.minibatch_base = 32; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}; sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = bmark_kimg # 12000

# Numerical precision (choose one).
desc += '-fp32'; sched.max_minibatch_per_gpu = {256: 8, 512: 4, 1024: 2}
#desc += '-fp16'; G.dtype = 'float16'; D.dtype = 'float16'; G.pixelnorm_epsilon=1e-4; G_opt.use_loss_scaling = True; D_opt.use_loss_scaling = True; sched.max_minibatch_per_gpu = {}

# Special modes.
#desc += '-VERBOSE'; sched.tick_kimg_base = 1; sched.tick_kimg_dict = {}; train.image_snapshot_ticks = 10; train.network_snapshot_ticks = 140;  train.resume_run_id = 15; train.resume_kimg = 6163
#desc += '-VERBOSE'; sched.tick_kimg_base = 1; sched.tick_kimg_dict = {}; train.image_snapshot_ticks = 1; train.network_snapshot_ticks = 100 # This is from karras' pro GAN, note it does not have resume_run_id
#desc += '-GRAPH'; train.save_tf_graph = True
#desc += '-HIST'; train.save_weight_histograms = True
