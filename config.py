import json
from pathlib import Path


class Config:
    """Config class

    Set using HPC to true in order to use appropriate paths for HPC
    """

    def __init__(self, tag, using_hpc):
        self.tag = tag

        if using_hpc:
            # HPC data dirs
            self.data_dirs_path = '/rds/general/user/aos13/home/HRTF-GANs-27Sep22-prep-for-publication'
            self.raw_hrtf_dir = Path('/rds/general/project/sonicom/live/HRTF Datasets')
        else:
            # local data dirs
            self.data_dirs_path = '/home/aos13/HRTF-GANs-27Sep22-prep-for-publication'
            self.raw_hrtf_dir = Path('/home/aos13/HRTF_datasets')

        self.path = f'{self.data_dirs_path}/runs/{self.tag}'
        self.model_path = f'{self.data_dirs_path}/runs/{self.tag}'

        self.train_hrtf_dir = self.data_dirs_path + '/data/train'
        self.valid_hrtf_dir = self.data_dirs_path + '/data/valid'
        self.train_hrtf_merge_dir = self.data_dirs_path + '/data/merge/train_merge'
        self.valid_hrtf_merge_dir = self.data_dirs_path + '/data/merge/valid_merge'
        self.mean_std_filename = self.data_dirs_path + '/data/mean_std_filename'

        # Data processing parameters
        self.merge_flag = True
        self.hrtf_size = 16
        self.upscale_factor = 4
        self.train_samples_ratio = 0.8

        # Training hyperparams
        self.batch_size = 8
        self.num_workers = 4
        self.num_epochs = 250  # was originally 250
        self.lr_gen = 0.0002
        self.lr_dis = 0.000001
        # how often to train the generator
        self.critic_iters = 4

        # Loss function weight
        self.content_weight = 1.0
        self.adversarial_weight = 0.1

        # betas for Adam optimizer
        self.beta1 = 0.9
        self.beta2 = 0.999

        self.ngpu = 1
        if self.ngpu > 0:
            self.device_name = "cuda:0"
        else:
            self.device_name = 'cpu'

    def save(self):
        j = {}
        for k, v in self.__dict__.items():
            j[k] = v
        with open(f'{self.path}/config.json', 'w') as f:
            json.dump(j, f)

    def load(self):
        with open(f'{self.path}/config.json', 'r') as f:
            j = json.load(f)
            for k, v in j.items():
                setattr(self, k, v)

    def get_train_params(self):
        return self.batch_size, self.beta1, self.beta2, self.num_epochs, self.lr_gen, self.lr_dis, self.critic_iters
