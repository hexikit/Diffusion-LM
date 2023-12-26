import numpy as np
from torch.utils.data import DataLoader, Dataset
import sys, os
import torch
import json

from outfit_catalogs import PolyvoreOutfitCatalog
from rq_vae import RQVAENew

def load_data_outfit(args):
    rq_vae = RQVAENew(args)
    if args.pretrained_rqvae:
        print("=> loading pretrained weights '{}'".format(args.pretrained_rqvae))
        checkpoint = torch.load(args.pretrained_rqvae, map_location=device)
        rq_vae.load_state_dict(checkpoint['state_dict'])
        print("=> loaded pretrained weights '{}'"
                .format(args.pretrained_rqvae))
    device_cpu = torch.device('cpu')
    rq_vae.to(device_cpu)

    vocab_size = rq_vae.codebook_n_levels * rq_vae.codebook_size
    special_tokens_dict = {
        'start_token_id': vocab_size,
        'end_token_id': vocab_size + 1,
        'item_start_token_id': vocab_size + 2,
        'pad_token_id': vocab_size + 3,
        'mask_token_id': vocab_size + 4,
        'retrieval_token_id': vocab_size + 5,
    }

    fn = os.path.join(args.datadir, 'polyvore_outfits', 'polyvore_item_metadata.json')
    meta_data = json.load(open(fn, 'r'))

    device = torch.device('cpu')

    outfit_catalog = PolyvoreOutfitCatalog(args, 'train', meta_data, device, rq_vae, special_tokens_dict)
    training_data = outfit_catalog.get_data()

    dataset = OutfitDataset(
        training_data,
        args
        # image_size,
        # data_args,
        # model_arch=data_args.model_arch,
    )
    return DataLoader(
        dataset,
        batch_size=64,  # 20,
        drop_last=True,
        shuffle=True,
        num_workers=1,
    )

class OutfitDataset(Dataset):
    def __init__(self, outfit_datasets, data_args, model_arch='conv-unet',
                 classes=None, shard=0, num_shards=1, mapping_func=None, model_emb=None):
        super().__init__()
        self.outfit_datasets = outfit_datasets
        self.length = len(self.outfit_datasets)
        self.model_arch = model_arch
        self.data_args = data_args
        self.mapping_func = mapping_func
        self.model_emb = model_emb
        # self.local_images = image_paths[shard:][::num_shards]
        # self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        arr = np.array(self.outfit_datasets[idx]['hidden_states'],
                        dtype=np.float32)
            
        if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
            # print(arr.dtype)
            # print(self.data_args.noise_level, 'using the noise level.')
            arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
            # print(arr.dtype)

        out_dict = {}
        out_dict['input_ids'] = np.array(self.outfit_datasets[idx]['input_ids'])
        # out_dict['mapping_func'] = self.mapping_func
        if self.data_args.experiment_mode == 'conditional_gen':
            out_dict['src_ids'] = np.array(self.outfit_datasets['train'][idx]['src_ids'])
            out_dict['src_mask'] = np.array(self.outfit_datasets['train'][idx]['src_mask'])
        # if self.local_classes is not None:
        #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return arr, out_dict
        # print(arr.dtype)
        # arr = arr.float()
        # print(arr.shape)