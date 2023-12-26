from __future__ import print_function
from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import json
import torch
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import faiss


def build_index(embeddings):
    # Load embeddings
    embeddings = embeddings.cpu().numpy().astype('float32')

    N, embed_dim = embeddings.shape

    # Create the index
    index = faiss.IndexFlatIP(embed_dim) 
    index.add(embeddings)
    return index

def search(index, query, k):
    """
    Retrieve k nearest neighbors for the given query.

    Parameters:
    - index: The FAISS index to search
    - query: A vector in the same embedding space (PyTorch tensor).
    - k: Number of neighbors to retrieve.

    Returns:
    - distances: Distances of the nearest neighbors.
    - indices: Indices of the nearest neighbors in the dataset.
    """
    query = query.numpy().astype('float32').reshape(1, -1)  # Convert the query to the right format
    distances, indices = index.search(query, k)
    return distances, indices

def compute_text_embed(sentences, fclip_model, fclip_processor, max_txt_len):
    sentences = [sent.strip() for sent in sentences]
    inputs = fclip_processor(text=sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_txt_len)
    with torch.no_grad():
        text_embeds = fclip_model.get_text_features(**inputs)
    
    normalized_text_embeds = F.normalize(text_embeds, p=2, dim=1)
    return normalized_text_embeds

def cosine_similarity(key_vectors: np.ndarray, space_vectors: np.ndarray, normalize=True):
    if normalize:
        key_vectors = key_vectors / np.linalg.norm(key_vectors, ord=2, axis=-1, keepdims=True)
    return np.matmul(key_vectors, space_vectors.T)

def zero_shot_classification(image_vectors, text_vectors, text_labels):
    # compute cosine similarity
    cosine_sim = cosine_similarity(image_vectors, text_vectors)

    preds = np.argmax(cosine_sim, axis=-1)
    return [text_labels[idx] for idx in preds]

def classify_item_embeds(txt_enc_model, item_idx_to_img_embed, max_txt_len):
    fclip_processor = CLIPProcessor.from_pretrained(txt_enc_model)
    fclip_model = CLIPModel.from_pretrained(txt_enc_model).eval()

    item_classifiers_embeds = compute_text_embed(ITEM_CLASSIFIERS_TEXT, fclip_model, fclip_processor, max_txt_len).numpy()

    image_vectors = item_idx_to_img_embed.clone().numpy()
    text_vectors = item_classifiers_embeds
    text_labels = ITEM_CLASSIFIERS_TEXT
    item_class_labels = zero_shot_classification(image_vectors, text_vectors, text_labels)
    return item_class_labels

def get_invalid_item_idxs(invalid_class_labels, class_indices):
    invalid_item_idxs_set = set()
    for invalid_class in invalid_class_labels:
        invalid_item_idxs = class_indices[invalid_class]
        invalid_item_idxs_set.update(invalid_item_idxs)
    return invalid_item_idxs_set

def get_invalid_outfit_ids(invalid_item_idxs_set, outfit_id_to_item_idxs):
    invalid_outfit_ids_set = set()
    for outfit_id, item_idxs in outfit_id_to_item_idxs.items():
        for item_idx in item_idxs:
            if item_idx in invalid_item_idxs_set:
                invalid_outfit_ids_set.add(outfit_id)
    return invalid_outfit_ids_set

def default_image_loader(path):
    return Image.open(path)

def decode_text(text):
    return text.replace('\n','').encode('ascii', 'ignore').strip().decode('ascii')

def get_item_description(item):    
    desc = ''
    if item.get('title', '').strip():
        return item['title'].strip()
    if item.get('url_name', '').strip():
        return item['url_name'].strip()
    if item.get('semantic_category', '').strip():
        return item['semantic_category'].strip()
    return desc

def get_outfit_description(outfit):
    if outfit.get('title', '').strip():
        desc = outfit['title'].strip()
        desc_aug = outfit['title'].strip()
        if 'outfit' in desc or 'Outfit' in desc:
            desc_aug + ' ' + 'Outfit'
        return desc, desc_aug
    
    if outfit.get('url_name', '').strip():
        desc = outfit['url_name'].strip()
        desc_aug = outfit['url_name'].strip()
        if 'outfit' in desc or 'Outfit' in desc:
            desc_aug + ' ' + 'outfit'
        return desc, desc_aug
    return '', ''


def get_z_hat_from_codes(all_codes):
    # all_codes: (n_levels, batch_size, embed_dim)
    reshaped_all_codes = all_codes.permute(1, 0, 2)
    return reshaped_all_codes.sum(dim=1)

def get_bag_of_words_set(item_idx_to_quantized_code):
    item_idx_to_quantized_code_cloned = item_idx_to_quantized_code.clone()
    item_idx_to_quantized_code_flat = item_idx_to_quantized_code_cloned.flatten()

    vocab = torch.unique(item_idx_to_quantized_code_flat)
    vocab_list = vocab.tolist()
    return set(vocab_list)

class PolyvoreItemCatalog():
    def __init__(self, args, split, meta_data, device, max_txt_len=77, transform=None, loader=default_image_loader):
        # self.model_artifacts_dir = os.path.join(args.outputdir, args.name)
        self.outfit_token = 0
        self.txt_embed_dim = args.txt_embed_dim
        self.item_embed_dim = args.item_embed_dim
        self.seed = int(args.seed)
        # Set seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.rootdir = os.path.join(args.datadir, 'polyvore_outfits', args.polyvore_split)
        self.impath = os.path.join(args.datadir, 'polyvore_outfits', 'images')
        self.is_train = split == 'train'
        # self.save_dir = f"{args.outputdir}/{args.name}"
        data_json = os.path.join(self.rootdir, '%s.json' % split)
        outfit_data = json.load(open(data_json, 'r'))
        print(f"Loading {split} dataset: {data_json}")

        # get list of images and make a mapping used to quickly organize the data
        # set of outfit ids
        outfit_ids = set()
        item_ids = set()
        outfit_id_to_item_ids = defaultdict(list)
        outfit_id_idx_to_item_id = {}

        item_id_to_category = {}
        item_id_to_fine_grain_category = {}
        category_to_item_ids = defaultdict(list)
        fine_grain_category_to_item_ids = defaultdict(list)
        self.outfit_id_to_item_category_cnt = defaultdict(lambda: defaultdict(int))

        num_outfits = 0
        max_items = 0
        for outfit in outfit_data:
            outfit_id = outfit['set_id']
            outfit_ids.add(outfit_id)
            num_outfits += 1
            
            items = outfit['items']
            cnt = len(items)
            max_items = max(cnt, max_items)

            for item in items:
                item_id = item['item_id']
                idx = str(item['index'])

                category = meta_data[item_id]['semantic_category']
                fine_grain_category = meta_data[item_id]['category_id']

                item_id_to_category[item_id] = category
                item_id_to_fine_grain_category[item_id] = fine_grain_category
                category_to_item_ids[category].append(item_id)
                fine_grain_category_to_item_ids[fine_grain_category].append(item_id)

                outfit_id_to_item_ids[outfit_id].append(item_id)
                outfit_id_idx_to_item_id[f"{outfit_id}_{idx}"] = item_id
                item_ids.add(item_id)
                self.outfit_id_to_item_category_cnt[outfit_id][category] += 1

        print(f"Num outfits in metadata file: {num_outfits}, {len(outfit_ids)}")
        print(f"Max items in outfit: {max_items}")

        # ["222049137", "222049138", ...]
        self.outfit_ids = sorted(list(outfit_ids))
        # ["159402796", "200810877", "208881110", ...]"
        self.item_ids = sorted(list(item_ids))
        # {"222049137": ["159402796", "200810877", "208881110", ...]}
        self.outfit_id_to_item_ids = outfit_id_to_item_ids
        # {"222049137_1": "159402796", ...}
        self.outfit_id_idx_to_item_id = outfit_id_idx_to_item_id
        self.max_items = max_items

        self.item_id_to_category = item_id_to_category
        self.item_id_to_fine_grain_category = item_id_to_fine_grain_category
        self.category_to_item_ids = category_to_item_ids
        self.fine_grain_category_to_item_ids = fine_grain_category_to_item_ids

        self.data = outfit_data
        self.transform = transform
        self.loader = loader
        self.split = split

        print(self.category_to_item_ids.keys())
        self.categories = sorted(list(self.category_to_item_ids.keys()))
        self.category_to_category_idx = {}
        for category_idx, category in enumerate(self.categories):
            self.category_to_category_idx[category] = category_idx


        self.item_idx_to_category = [''] * len(self.item_ids)
        self.item_idx_to_fine_grain_category = [''] * len(self.item_ids)
        self.item_idx_to_category_idx = [0] * len(self.item_ids)

        self.category_to_item_idxs = defaultdict(list)
        self.fine_grain_category_to_item_idxs = defaultdict(list)

        # ["red blazer", "pink_dress", ...]
        self.item_idx_to_desc = [''] * len(self.item_ids)
        # {"159402796": 0, "200810877": 1, "208881110": 2, ...}
        self.item_id_to_index = {}
        cnt = 0
        print(f"Items in meta_data: {len(meta_data.items())}")
        for idx, item_id in enumerate(self.item_ids):
            item = meta_data[item_id]
            item_desc = get_item_description(item)

            # item_desc = str(item_desc.replace('\n','').encode('ascii', 'ignore').strip().lower())
            item_desc = decode_text(item_desc)
            # print(item_desc)
            if len(item_desc) > 0: cnt += 1

            category = self.item_id_to_category[item_id]
            category_idx = self.category_to_category_idx[category]
            fine_grain_category = self.item_id_to_fine_grain_category[item_id]

            self.item_idx_to_category[idx] = category
            self.item_idx_to_fine_grain_category[idx] = fine_grain_category
            self.item_idx_to_category_idx[idx] = category_idx

            self.category_to_item_idxs[category].append(idx)
            self.fine_grain_category_to_item_idxs[fine_grain_category].append(idx)
            
            self.item_idx_to_desc[idx] = item_desc
            self.item_id_to_index[item_id] = idx
        print(f"There are [{cnt}/{len(self.item_ids)}] items with text descriptions")


        outfit_desc_cnt = 0
        outfit_desc_with_outfit_cnt = 0
        outfit_meta_data_fn = os.path.join(args.datadir, 'polyvore_outfits', 'polyvore_outfit_titles.json')
        outfit_meta_data = json.load(open(outfit_meta_data_fn, 'r'))

        # ["red blazer", "pink_dress", ...]
        self.outfit_idx_to_desc = [''] * len(self.outfit_ids)
        self.outfit_idx_to_desc_aug = [''] * len(self.outfit_ids)
        # {"159402796": 0, "200810877": 1, "208881110": 2, ...}
        self.outfit_id_to_index = {}

        title_desc_cnt = 0
        url_name_desc_cnt = 0
        title_desc_with_outfit_cnt = 0
        url_name_desc_with_outfit_cnt = 0
        self.outfit_idx_to_title_desc = [''] * len(self.outfit_ids)
        self.outfit_idx_to_url_name_desc = [''] * len(self.outfit_ids)
        # These are augmented descriptions with 'outfit' appended to the end
        self.outfit_idx_to_title_desc_aug = [''] * len(self.outfit_ids)
        self.outfit_idx_to_url_name_desc_aug = [''] * len(self.outfit_ids)
        self.outfit_idx_to_desc_list_len = [0] * len(self.outfit_ids)
        for idx, outfit_id in enumerate(self.outfit_ids):
            outfit = outfit_meta_data[outfit_id]
            outfit_desc, outfit_desc_aug = get_outfit_description(outfit)
            if 'outfit' in outfit_desc or 'Outfit' in outfit_desc: 
                outfit_desc_with_outfit_cnt += 1

            outfit_desc = decode_text(outfit_desc)
            if len(outfit_desc) > 0: outfit_desc_cnt += 1

            title_desc = ''
            url_name_desc = ''
            if outfit.get('title', '').strip():
                title_desc = decode_text(outfit['title'].strip())
    
            if outfit.get('url_name', '').strip():
                url_name_desc = decode_text(outfit['url_name'].strip())

            if len(title_desc) > 0: 
                self.outfit_idx_to_desc_list_len[idx] += 1
                title_desc_cnt += 1
                if 'outfit' in title_desc or 'Outfit' in title_desc: 
                    title_desc_with_outfit_cnt += 1
                else:
                    # print(title_desc + ' ' + 'Outfit')
                    self.outfit_idx_to_title_desc_aug[idx] = title_desc + ' ' + 'Outfit'
            if len(url_name_desc) > 0: 
                self.outfit_idx_to_desc_list_len[idx] += 1
                url_name_desc_cnt += 1
                if 'outfit' in url_name_desc or 'Outfit' in url_name_desc: 
                    url_name_desc_with_outfit_cnt += 1
                else:
                    # print(url_name_desc + ' ' + 'outfit')
                    self.outfit_idx_to_url_name_desc_aug[idx] = url_name_desc + ' ' + 'outfit'

            self.outfit_idx_to_title_desc[idx] = title_desc
            self.outfit_idx_to_url_name_desc[idx] = url_name_desc
            # self.outfit_idx_to_desc_list[idx] = [title_desc, url_name_desc]

            self.outfit_idx_to_desc[idx] = outfit_desc
            self.outfit_idx_to_desc_aug[idx] = outfit_desc_aug
            self.outfit_id_to_index[outfit_id] = idx
        print(f"There are [{outfit_desc_cnt}/{len(self.outfit_ids)}] outfits with text descriptions")
        print(f"There are [{title_desc_cnt}/{len(self.outfit_ids)}] outfits with title descriptions")
        print(f"There are [{url_name_desc_cnt}/{len(self.outfit_ids)}] outfits with url name descriptions")
        print(f"There are [{outfit_desc_with_outfit_cnt}/{len(self.outfit_ids)}] outfits with text descriptions that have 'outfit' in description")
        print(f"There are [{title_desc_with_outfit_cnt}/{len(self.outfit_ids)}] outfits with title descriptions that have 'outfit' in description")
        print(f"There are [{url_name_desc_with_outfit_cnt}/{len(self.outfit_ids)}] outfits with url name descriptions that have 'outfit' in description")

        self.outfit_id_to_item_idxs = {}
        for outfit_id, item_ids in self.outfit_id_to_item_ids.items():
            self.outfit_id_to_item_idxs[outfit_id] = [self.item_id_to_index[item_id] for item_id in item_ids]

        print(f"outfit_ids len: {len(self.outfit_ids)}... {self.outfit_ids[:8]}")
        print(f"items_ids len: {len(self.item_ids)}... {self.item_ids[:8]}")


        self.max_txt_len = max_txt_len
        self.txt_enc_model = args.txt_enc_model
        self.img_enc_model = args.img_enc_model

        item_embeds_dir = f"{args.embedsdir}"
        if not os.path.exists(item_embeds_dir):
            os.makedirs(item_embeds_dir)

        # Precomputing or loading item description embeddings
        txt_enc_model_name = self.txt_enc_model
        if '/' in txt_enc_model_name:
            txt_enc_model_name = txt_enc_model_name.split('/')[1]

        item_desc_embeds_fn = os.path.join(item_embeds_dir, f'{txt_enc_model_name}_item_desc_embeds_{split}.pkl')
        if not os.path.exists(item_desc_embeds_fn):
            self._precompute_desc_embeds(item_desc_embeds_fn, self.item_ids, self.item_idx_to_desc, args.cuda, args.batch_size)

        if os.path.exists(item_desc_embeds_fn):
            self.item_idx_to_desc_embed = torch.load(item_desc_embeds_fn, map_location=device).cpu()
            if args.norm_embeds: self.item_idx_to_desc_embed = self._normalize_embeddings(self.item_idx_to_desc_embed)
            assert len(self.item_idx_to_desc_embed) == len(self.item_ids)


        # Precomputing or loading outfit description embeddings
        outfit_desc_embeds_fn = os.path.join(item_embeds_dir, f'{txt_enc_model_name}_outfit_desc_embeds_{split}.pkl')
        if not os.path.exists(outfit_desc_embeds_fn):
            self._precompute_desc_embeds(outfit_desc_embeds_fn, self.outfit_ids, self.outfit_idx_to_desc, args.cuda, args.batch_size)

        if os.path.exists(outfit_desc_embeds_fn):
            self.outfit_idx_to_desc_embed = torch.load(outfit_desc_embeds_fn, map_location=device).cpu()
            if args.norm_embeds: self.outfit_idx_to_desc_embed = self._normalize_embeddings(self.outfit_idx_to_desc_embed)
            assert len(self.outfit_idx_to_desc_embed) == len(self.outfit_ids)

        # Precomputing or loading augmented outfit description embeddings
        outfit_desc_aug_embeds_fn = os.path.join(item_embeds_dir, f'{txt_enc_model_name}_outfit_desc_aug_embeds_{split}.pkl')
        if not os.path.exists(outfit_desc_aug_embeds_fn):
            self._precompute_desc_embeds(outfit_desc_aug_embeds_fn, self.outfit_ids, self.outfit_idx_to_desc_aug, args.cuda, args.batch_size)

        if os.path.exists(outfit_desc_aug_embeds_fn):
            self.outfit_idx_to_desc_aug_embed = torch.load(outfit_desc_aug_embeds_fn, map_location=device).cpu()
            if args.norm_embeds: self.outfit_idx_to_desc_aug_embed = self._normalize_embeddings(self.outfit_idx_to_desc_aug_embed)
            assert len(self.outfit_idx_to_desc_aug_embed) == len(self.outfit_ids)

        
        # Precomputing or loading outfit title description embeddings
        outfit_title_desc_embeds_fn = os.path.join(item_embeds_dir, f'{txt_enc_model_name}_outfit_title_desc_embeds_{split}.pkl')
        if not os.path.exists(outfit_title_desc_embeds_fn):
            self._precompute_desc_embeds(outfit_title_desc_embeds_fn, self.outfit_ids, self.outfit_idx_to_title_desc, args.cuda, args.batch_size)

        if os.path.exists(outfit_title_desc_embeds_fn):
            self.outfit_idx_to_title_desc_embed = torch.load(outfit_title_desc_embeds_fn, map_location=device).cpu()
            if args.norm_embeds: self.outfit_idx_to_title_desc_embed = self._normalize_embeddings(self.outfit_idx_to_title_desc_embed)
            assert len(self.outfit_idx_to_title_desc_embed) == len(self.outfit_ids)

        # Precomputing or loading augmented outfit title description embeddings
        outfit_title_desc_aug_embeds_fn = os.path.join(item_embeds_dir, f'{txt_enc_model_name}_outfit_title_desc_aug_embeds_{split}.pkl')
        if not os.path.exists(outfit_title_desc_aug_embeds_fn):
            self._precompute_desc_embeds(outfit_title_desc_aug_embeds_fn, self.outfit_ids, self.outfit_idx_to_title_desc_aug, args.cuda, args.batch_size)

        if os.path.exists(outfit_title_desc_aug_embeds_fn):
            self.outfit_idx_to_title_desc_aug_embed = torch.load(outfit_title_desc_aug_embeds_fn, map_location=device).cpu()
            if args.norm_embeds: self.outfit_idx_to_title_desc_aug_embed = self._normalize_embeddings(self.outfit_idx_to_title_desc_aug_embed)
            assert len(self.outfit_idx_to_title_desc_aug_embed) == len(self.outfit_ids)


        # Precomputing or loading outfit url name description embeddings
        outfit_url_name_desc_embeds_fn = os.path.join(item_embeds_dir, f'{txt_enc_model_name}_outfit_url_name_desc_embeds_{split}.pkl')
        if not os.path.exists(outfit_url_name_desc_embeds_fn):
            self._precompute_desc_embeds(outfit_url_name_desc_embeds_fn, self.outfit_ids, self.outfit_idx_to_url_name_desc, args.cuda, args.batch_size)

        if os.path.exists(outfit_url_name_desc_embeds_fn):
            self.outfit_idx_to_url_name_desc_embed = torch.load(outfit_url_name_desc_embeds_fn, map_location=device).cpu()
            if args.norm_embeds: self.outfit_idx_to_url_name_desc_embed = self._normalize_embeddings(self.outfit_idx_to_url_name_desc_embed)
            assert len(self.outfit_idx_to_url_name_desc_embed) == len(self.outfit_ids)
        
        # Precomputing or loading augmented outfit url name description embeddings
        outfit_url_name_desc_aug_embeds_fn = os.path.join(item_embeds_dir, f'{txt_enc_model_name}_outfit_url_name_desc_aug_embeds_{split}.pkl')
        if not os.path.exists(outfit_url_name_desc_aug_embeds_fn):
            self._precompute_desc_embeds(outfit_url_name_desc_aug_embeds_fn, self.outfit_ids, self.outfit_idx_to_url_name_desc_aug, args.cuda, args.batch_size)

        if os.path.exists(outfit_url_name_desc_aug_embeds_fn):
            self.outfit_idx_to_url_name_desc_aug_embed = torch.load(outfit_url_name_desc_aug_embeds_fn, map_location=device).cpu()
            if args.norm_embeds: self.outfit_idx_to_url_name_desc_aug_embed = self._normalize_embeddings(self.outfit_idx_to_url_name_desc_aug_embed)
            assert len(self.outfit_idx_to_url_name_desc_aug_embed) == len(self.outfit_ids)


        # Precomputing or loading image embeddings
        img_enc_model_name = self.img_enc_model
        if '/' in img_enc_model_name:
            img_enc_model_name = img_enc_model_name.split('/')[1]

        item_img_embeds_fn = os.path.join(item_embeds_dir, f'{img_enc_model_name}_item_img_embeds_{split}.pkl')
        if not os.path.exists(item_img_embeds_fn):
            self._precompute_img_embeds(self.img_enc_model, item_img_embeds_fn, self.item_ids, self.impath, self.loader, self.transform, args.cuda, args.batch_size)
        
        if os.path.exists(item_img_embeds_fn):
            self.item_idx_to_img_embed = torch.load(item_img_embeds_fn, map_location=device).cpu()
            self.item_idx_to_img_embed_unnorm = self.item_idx_to_img_embed.clone()
            if args.norm_embeds: self.item_idx_to_img_embed = self._normalize_embeddings(self.item_idx_to_img_embed)
            assert len(self.item_idx_to_img_embed) == len(self.item_ids)

        
        # Precomputing or loading category description embeddings
        category_desc_embeds_fn = os.path.join(item_embeds_dir, f'{txt_enc_model_name}_category_desc_embeds_{split}.pkl')
        self.catgories = sorted(self.category_to_item_ids.keys())
        self.category_to_category_idx = {category: idx for idx, category in enumerate(self.catgories)}
        if not os.path.exists(category_desc_embeds_fn):
            category_desc = [f"clothing {category}" for category in self.categories]
            self._precompute_desc_embeds(category_desc_embeds_fn, category_desc, category_desc, args.cuda, args.batch_size)

        if os.path.exists(category_desc_embeds_fn):
            self.category_idx_to_category_desc_embed = torch.load(category_desc_embeds_fn, map_location=device).cpu()
            if args.norm_embeds: self.category_idx_to_category_desc_embed = self._normalize_embeddings(self.category_idx_to_category_desc_embed)
            assert len(self.category_idx_to_category_desc_embed) == len(self.categories)

        self.item_idx_to_category_desc_embed = torch.zeros((len(self.item_ids), args.txt_embed_dim))
        for item_idx, item_id in enumerate(self.item_ids):
            category = self.item_id_to_category[item_id]
            category_idx = self.category_to_category_idx[category]

            self.item_idx_to_category_desc_embed[item_idx, :] = self.category_idx_to_category_desc_embed[category_idx, :].clone()

        

        self.outfit_id_to_item_category_embeds = {}
        for outfit_id in self.outfit_ids:
            outfit_item_idxs = self.outfit_id_to_item_idxs[outfit_id]
            self.outfit_id_to_item_category_embeds[outfit_id] = torch.cat([self.item_idx_to_category_desc_embed[item_idx].clone().unsqueeze(0) for item_idx in outfit_item_idxs], dim=0)

        # if self.pca:
        #     print("Applying PCA reduction")
        #     self.outfit_idx_to_pca_reduced_embed = self.pca_transform_outfit_embeds(self.outfit_idx_to_embed.clone().cpu().numpy())

        self.fine_grain_category_to_item_embeds = {}
        # Adds a layer of indirection where we take the index of items in fine grain category and map it back to the original item index
        self.fine_grain_category_to_item_embeds_to_item_idxs = {}
        for fine_grain_category, items_idxs in self.fine_grain_category_to_item_idxs.items():
            items_idxs_tensor = torch.tensor(items_idxs, dtype=torch.int)
            self.fine_grain_category_to_item_embeds[fine_grain_category] = self.item_idx_to_img_embed[items_idxs_tensor]
            self.fine_grain_category_to_item_embeds_to_item_idxs[fine_grain_category] = items_idxs_tensor

        self.category_to_item_embeds = {}
        # Adds a layer of indirection where we take the index of items in fine grain category and map it back to the original item index
        self.category_to_item_embeds_to_item_idxs = {}
        for category, items_idxs in self.category_to_item_idxs.items():
            items_idxs_tensor = torch.tensor(items_idxs, dtype=torch.int)
            self.category_to_item_embeds[category] = self.item_idx_to_img_embed[items_idxs_tensor]
            self.category_to_item_embeds_to_item_idxs[category] = items_idxs_tensor

        self.category_to_embed_index = {}
        for category, item_embeds in self.category_to_item_embeds.items():
            self.category_to_embed_index[category] = build_index(item_embeds)

        self.fine_grain_category_to_embed_index = {}
        for fine_grain_category, item_embeds in self.fine_grain_category_to_item_embeds.items():
            self.fine_grain_category_to_embed_index[fine_grain_category] = build_index(item_embeds)
        
        print(f"Finished loading {split} base dataset\n")
    
    def _normalize_embeddings(self, embeddings):
        # print(f"Normalizing precomputed item embeddings")
        # print(f"Before normalizing: {embeddings[0, :8]}")
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        # print(f"After normalizing: {normalized_embeddings[0, :8]}")
        return normalized_embeddings


    def _get_invalid_item_idxs(self):
        # fclip_processor = CLIPProcessor.from_pretrained(self.txt_enc_model)
        # fclip_model = CLIPModel.from_pretrained(self.txt_enc_model).eval()

        # item_classifiers_embeds = compute_text_embed(ITEM_CLASSIFIERS_TEXT, fclip_model, fclip_processor, self.max_txt_len).numpy()

        # image_vectors = self.item_idx_to_img_embed.clone().numpy()
        # text_vectors = item_classifiers_embeds
        # text_labels = ITEM_CLASSIFIERS_TEXT
        # item_class_labels = zero_shot_classification(image_vectors, text_vectors, text_labels)
        item_class_labels = classify_item_embeds(self.txt_enc_model, self.item_idx_to_img_embed, self.max_txt_len)
        # print(item_class_labels)

        item_class_indices = defaultdict(list)
        for index, label in enumerate(item_class_labels):
            item_class_indices[label].append(index)

        invalid_item_idxs_set = get_invalid_item_idxs(INVALID_ITEM_CLASSES, item_class_indices)
        # print(f"Saving precomputed invalid item idxs {invalid_item_idxs_fn}")
        # torch.save(torch.tensor(list(invalid_item_idxs_set), dtype=torch.long).cpu(), invalid_item_idxs_fn)
        return invalid_item_idxs_set

    def _load_image(self, item_id):
        img_path = os.path.join(self.impath, f"{str(item_id)}.jpg")
        return Image.open(img_path)
    
    def load_item(self, item_idx):
        item_idx_tensor = torch.tensor(item_idx, dtype=torch.int)
        return self.item_idx_to_img_embed[item_idx_tensor]



def reorder_outfits(item_id_to_l1_code, outfit_id_to_item_ids_list):
    # Function to get priority of an item ID
    def get_priority(item_id):
        category_index = item_id_to_l1_code[item_id]
        return CATEGORY_INDEX_TO_PRIORITY.get(category_index, float('inf'))

    # Sort item IDs in each outfit based on priority
    sorted_outfits = {}
    for outfit_id, item_ids in outfit_id_to_item_ids_list.items():
        sorted_outfits[outfit_id] = sorted(item_ids, key=get_priority)

    return sorted_outfits

class PolyvoreOutfitCatalog(PolyvoreItemCatalog):
    def __init__(self, args, split, meta_data, device, outfit_tokenizer_model, special_tokens_dict, special_tokens_to_embs, max_txt_len=77, transform=None, loader=default_image_loader, pretrain=True):
        super().__init__(args, split, meta_data, device, max_txt_len, transform, loader)
        self.special_tokens_dict = special_tokens_dict
        self.special_tokens_to_embs = special_tokens_to_embs

        self.bos_token_id = special_tokens_dict['start_token_id']
        self.eos_token_id = special_tokens_dict['end_token_id']
        self.item_start_token_id = special_tokens_dict['item_start_token_id']
        self.pad_token_id = special_tokens_dict['pad_token_id']
        self.mask_token_id = special_tokens_dict['mask_token_id']
        self.retrieval_token_id = special_tokens_dict['retrieval_token_id']


        # self.use_retrieval_token = args.use_retrieval_token
        # if self.use_retrieval_token:
        #     print(f"Training w/ retrieval token")

        self.codebook_n_levels = outfit_tokenizer_model.codebook_n_levels
        self.vocab_size = outfit_tokenizer_model.codebook_n_levels * outfit_tokenizer_model.codebook_size
        self.vocab_size += len(special_tokens_dict)

        self.outfit_min_items = args.outfit_min_items
        self.outfit_max_items = args.outfit_max_items

        # self.debug = args.debug
        # self.item_start_token_beginning = args.item_start_token_beginning
        # self.use_special_tokens_pred = args.use_special_tokens_pred
        # self.randomize_item_pos = args.randomize_item_pos
        # if self.randomize_item_pos:
        #     print("Training with randomized item positions")

        # Precomputing or loading category description embeddings
        item_embeds_dir = f"{args.embedsdir}"
        if not os.path.exists(item_embeds_dir):
            os.makedirs(item_embeds_dir)

        invalid_item_idxs_fn = os.path.join(item_embeds_dir, f'item_classification_{split}.pkl')
        if not os.path.exists(invalid_item_idxs_fn):
            self._get_invalid_item_idxs(invalid_item_idxs_fn)
            # self._construct_outfit_split_set()

        if os.path.exists(invalid_item_idxs_fn):
            invalid_item_idxs_tensor = torch.load(invalid_item_idxs_fn, map_location=device).cpu()

            invalid_item_idxs_set = set(invalid_item_idxs_tensor.tolist())
            self.split_set_outfit_ids = self._construct_outfit_split_set(invalid_item_idxs_set, self.outfit_min_items, self.outfit_max_items)
        
        # if pretrain:
        #     self.foo()
        #     self.split_set_outfit_ids = []
        #     for outfit_id in self.split_set_outfit_ids_to_partial_outfit_items.keys():
        #         self.split_set_outfit_ids.append(outfit_id)

        # self.split_set_outfit_ids = self._construct_outfit_split_set()
        
        n_shuffles = 3
        for _ in range(n_shuffles):
            self.shuffle()

        self.min_split_set_outfit_len = float('inf')
        self.max_split_set_outfit_len = float('-inf')
        self.total_split_set_outfit_len = 0
        for outfit_id in self.split_set_outfit_ids:
            # outfit_index = self.outfit_id_to_index[outfit_id]
            outfit_items_idxs = self.outfit_id_to_item_idxs[outfit_id]
            outfit_len = len(outfit_items_idxs)

            self.min_split_set_outfit_len = min(self.min_split_set_outfit_len, outfit_len)
            self.max_split_set_outfit_len = max(self.max_split_set_outfit_len, outfit_len)
            self.total_split_set_outfit_len += outfit_len

        self.avg_split_set_outfit_len = self.total_split_set_outfit_len / len(self.split_set_outfit_ids)

        print(f"Min outfit len: {self.min_split_set_outfit_len}")
        print(f"Max outfit len: {self.max_split_set_outfit_len}")
        print(f"Avg outfit len: {self.avg_split_set_outfit_len}")
        print(f"N outfits in split set: {len(self.split_set_outfit_ids)}")

        item_embeds_dir = f"{args.embedsdir}"
        if not os.path.exists(item_embeds_dir):
            os.makedirs(item_embeds_dir)

        # Quantizing items
        rq_vae_model_split_path = args.pretrained_rqvae.split('/')
        rq_vae_model_name = rq_vae_model_split_path[-2]
        quantized_item_codes_fn = os.path.join(item_embeds_dir, f'{rq_vae_model_name}_quantized_item_codes_{split}.pkl')
        quantized_item_code_embs_fn = os.path.join(item_embeds_dir, f'{rq_vae_model_name}_quantized_item_code_embs_{split}.pkl')
        quantized_item_embeds_fn = os.path.join(item_embeds_dir, f'{rq_vae_model_name}_quantized_item_embeds_{split}.pkl')
        if not os.path.exists(quantized_item_codes_fn):
            self._quantize_items(quantized_item_codes_fn, quantized_item_code_embs_fn, quantized_item_embeds_fn, outfit_tokenizer_model, self.item_ids, map_to_merged_codebook=True, batch_size=args.batch_size)

        if os.path.exists(quantized_item_codes_fn):
            self.item_idx_to_quantized_code = torch.load(quantized_item_codes_fn, map_location=device).cpu()
            self.item_idx_to_quantized_code_embs = torch.load(quantized_item_code_embs_fn, map_location=device).cpu()
            self.item_idx_to_quantized_embed = torch.load(quantized_item_embeds_fn, map_location=device).cpu()
            # if args.norm_embeds: self.outfit_idx_to_url_name_desc_embed = self._normalize_embeddings(self.outfit_idx_to_url_name_desc_embed)
            self.split_set_bag_of_words = get_bag_of_words_set(self.item_idx_to_quantized_code)
            print(f"N unique tokens: {len(self.split_set_bag_of_words)}")

            assert len(self.item_idx_to_quantized_code) == len(self.item_ids)
            assert len(self.item_idx_to_quantized_code_embs) == len(self.item_ids)
            assert len(self.item_idx_to_quantized_embed) == len(self.item_ids)

        self.item_quantized_codes_to_item_ids = defaultdict(list)
        self.item_quantized_code_to_item_ids_prefix_tree = defaultdict(set)
        self.item_id_to_l1_code = {}
        self.item_idx_to_l1_code = [0] * len(self.item_ids)
        self.quantized_category_index_to_item_codes = defaultdict(list)
        self.split_set_quantized_category_index_to_item_codes = defaultdict(list)
        item_start_token_code_tuple = (self.item_start_token_id,)
        for i in range(len(self.item_idx_to_quantized_code)):
            item_quantized_code = tuple(self.item_idx_to_quantized_code[i].tolist())
            item_id = self.item_ids[i]
            category_index = item_quantized_code[0]

            self.item_id_to_l1_code[item_id] = category_index
            self.item_idx_to_l1_code[i] = category_index
            self.quantized_category_index_to_item_codes[category_index].append(item_quantized_code)

            self.item_quantized_codes_to_item_ids[item_quantized_code].append(item_id)
            for j in range(len(item_quantized_code) - 1):
                self.item_quantized_code_to_item_ids_prefix_tree[item_quantized_code[:j+1]].add(item_quantized_code[j+1])
            # Mapping item start token to list of category tokens
            # print(item_quantized_code)
            self.item_quantized_code_to_item_ids_prefix_tree[item_start_token_code_tuple].add((item_quantized_code[0],))

            if i not in invalid_item_idxs_set:
                self.split_set_quantized_category_index_to_item_codes[category_index].append(item_quantized_code)


        self.split_set_outfit_id_to_sorted_item_idxs = {}
        self.split_set_outfit_len_to_outfit_ids = defaultdict(list)
        for outfit_id in self.split_set_outfit_ids:
            def get_priority_item_idx(item_idx):
                category_index = self.item_idx_to_l1_code[item_idx]
                return CATEGORY_INDEX_TO_PRIORITY.get(category_index, float('inf'))
            
            outfit_item_idxs = self.outfit_id_to_item_idxs[outfit_id]
            self.split_set_outfit_id_to_sorted_item_idxs[outfit_id] = sorted(outfit_item_idxs, key=get_priority_item_idx)

            outfit_len = len(outfit_item_idxs)
            self.split_set_outfit_len_to_outfit_ids[outfit_len].append(outfit_id) 

            # tokenized_outfit, outfit_item_seqmentation = self._tokenize_outfit(outfit_item_idxs, self.max_split_set_outfit_len)

            

        # Quantize all items to get tuples and their quantized embeds for FITB
        # if self.is_train:
        #     all_tokenized_outfits = []
        #     all_outfit_item_segmentation = []
        #     for outfit_id in self.split_set_outfit_ids:
        #         tokenized_outfit_sequence, outfit_item_segmentation = self._construct_outfit_item_codes(outfit_id, self.max_split_set_outfit_len)
        #         all_tokenized_outfits.append(tokenized_outfit_sequence)
        #         all_outfit_item_segmentation.append(outfit_item_segmentation)
        #     self.split_set_tokenized_outfits = torch.stack(all_tokenized_outfits)
        #     self.split_set_outfit_item_segmentation = torch.stack(all_outfit_item_segmentation)

        print(f"Finished loading {split} outfit dataset\n")

    def get_data(self):
        data = []
        for index in range(len(self.split_set_outfit_ids)):
            outfit_id = self.split_set_outfit_ids[index]
            outfit_items_idxs = self.outfit_id_to_item_idxs[outfit_id]

            tokenized_outfit_item_codes, tokenized_outfit_item_code_embs = self._tokenize_outfit(outfit_items_idxs)

            current_length = tokenized_outfit_item_codes.shape[0]
            max_sequence_length = self.max_split_set_outfit_len * (self.codebook_n_levels + 1)
            # print(f"current_length: {current_length}")
            # print(f"max_sequence_length: {max_sequence_length}")

            padding_size = max_sequence_length - current_length
            # print(f"padding_size: {padding_size}")
            input_ids = F.pad(tokenized_outfit_item_codes, pad=(0, padding_size), value=self.pad_token_id)

            # latent_dim = tokenized_outfit_item_code_embs.shape[-1]
            # pad_token_embs = torch.randn(padding_size, latent_dim)
            pad_token_emb = self.special_tokens_to_embs[self.pad_token_id].clone()
            pad_token_embs = pad_token_emb.repeat(padding_size, 1)
            hidden_states = torch.cat((tokenized_outfit_item_code_embs, pad_token_embs))

            attention_mask = (input_ids != self.pad_token_id).bool()
            target_mask = torch.zeros_like(input_ids).bool()
            for special_token_id in self.special_tokens_dict.values():
                target_mask |= (input_ids == special_token_id)

            sample = {
                "input_ids": input_ids.numpy(),
                "hidden_states": hidden_states.numpy(),
                'attention_mask': attention_mask.numpy(),
                'target_mask': target_mask.numpy(),
            }
            data.append(sample)
        return data
    


    def _construct_outfit_split_set(self, invalid_item_idxs_set, outfit_min_items, outfit_max_items):
        invalid_outfit_ids_set = get_invalid_outfit_ids(invalid_item_idxs_set, self.outfit_id_to_item_idxs)

        valid_outfit_ids = []
        for outfit_id in self.outfit_ids:
            if outfit_id not in invalid_outfit_ids_set:
                outfit_item_idxs = self.outfit_id_to_item_idxs[outfit_id]
                outfit_len = len(outfit_item_idxs)
                if outfit_len >= outfit_min_items and outfit_len <= outfit_max_items:
                    n_accessories = 0
                    for item_idx in outfit_item_idxs:
                        category = self.item_idx_to_category[item_idx]
                        if category in ['jewellery', 'bags', 'hats', 'accessories', 'sunglasses', 'scarves']:
                            n_accessories += 1
                    if n_accessories < outfit_len:
                        valid_outfit_ids.append(outfit_id)
        return valid_outfit_ids
    
    def _map_semantic_ids_to_merged_codebook(self, semantic_ids_tensor, codebook_size):
        batch_size, n_levels = semantic_ids_tensor.shape
        # Calculate the offset for each level's codewords
        offsets = (torch.arange(n_levels) * codebook_size).unsqueeze(0).repeat(batch_size, 1)
        # Map original codewords to the new indices in the merged codebook
        new_codewords = semantic_ids_tensor + offsets
        # Flatten the codewords if necessary or leave them as tuples, depending on your downstream use
        return new_codewords
    
    def _quantize_items(self, quantized_item_codes_fn, quantized_item_code_embs_fn, quantized_item_embeds_fn, model, item_ids, map_to_merged_codebook=True, batch_size=64):
        print(f"Quantizing items")
        N = len(item_ids)
        codebook_n_levels = model.codebook_n_levels
        codebook_size = model.codebook_size

        quantized_item_codes = []
        quantized_item_code_embs = []
        quantized_item_embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, N, batch_size), desc="Processing batches"):
                item_img_embeds = self.item_idx_to_img_embed[i: i+batch_size].clone()
                batch_len = len(item_img_embeds)

                # x_hat, z_hat, all_residuals, all_quantized_embeds, item_codes_list = model(item_img_embeds)
                x_hat, indices, commit_loss, all_codes = model(item_img_embeds.unsqueeze(1), return_all_codes=True)
                # z_hat = model.get_output_from_indices(indices).squeeze(1)
                # print(f"all_codes: {all_codes.shape}")
                all_codes = all_codes.squeeze(2)
                indices = indices.squeeze(1)
                batch_len = indices.shape[0]

                z_hat = get_z_hat_from_codes(all_codes)

                # Update codebook usage statistics
                item_img_embed_codes_tensor = torch.zeros(batch_len, codebook_n_levels, dtype=torch.long)
                for level in range(codebook_n_levels):
                    indices_level = indices[:, level].cpu()

                    item_img_embed_codes_tensor[:, level] = indices_level

                if map_to_merged_codebook:
                    items_codes_tensor = self._map_semantic_ids_to_merged_codebook(item_img_embed_codes_tensor, codebook_size)
                # print(f"items_codes_tensor: {items_codes_tensor.shape}")
                all_codes_permuted = all_codes.permute(1, 0, 2)
                # print(f"all_codes_permuted: {all_codes_permuted.shape}")

                quantized_item_codes.append(items_codes_tensor)
                quantized_item_code_embs.append(all_codes_permuted)
                quantized_item_embeds.append(z_hat)

        # Concatenate the batches of embeddings
        quantized_item_codes = torch.cat(quantized_item_codes, dim=0)
        quantized_item_code_embs = torch.cat(quantized_item_code_embs, dim=0)
        quantized_item_embeds = torch.cat(quantized_item_embeds, dim=0)

        # Save the embeddings
        print(f"Saving quantized item codes to {quantized_item_codes_fn}")
        torch.save(quantized_item_codes.cpu(), quantized_item_codes_fn)

        print(f"Saving quantized item code embs to {quantized_item_code_embs_fn}")
        torch.save(quantized_item_code_embs.cpu(), quantized_item_code_embs_fn)

        print(f"Saving quantized item codes to {quantized_item_embeds_fn}")
        torch.save(quantized_item_embeds.cpu(), quantized_item_embeds_fn)

    def outfit_item_codes_to_outfit_item_ids(self, outfit_item_codes):
        # Shape: (n_items, item_seq_len)
        # List of tuples
        # outfit_item_ids = [self.item_quantized_codes_to_item_ids[tuple(item_code.tolist())][0] for item_code in outfit_item_codes]
        outfit_item_ids = []
        for item_code in outfit_item_codes:
            item_code_tuple = tuple(item_code.tolist())
            print(f"item_code_tuple: {item_code_tuple}")
            item_ids_list = self.item_quantized_codes_to_item_ids[tuple(item_code.tolist())]
            print(f"item_ids_list: {item_ids_list}")
            if len(item_ids_list) == 0:
                code_len = len(item_code)
                for i in range(1, code_len):
                    next_tokens = list(self.item_quantized_code_to_item_ids_prefix_tree[tuple(item_code.tolist())[:code_len-i]])
                    print(f"tuple(item_code.tolist())[:code_len-i]: {tuple(item_code.tolist())[:code_len-i]}")
                    print(f"next_tokens: {next_tokens}")
                    # outfit_item_ids.append(item_ids_list[0])
                    sampled_code = tuple(item_code.tolist())[:code_len-i] + (next_tokens[0],)
                    item_ids_list = self.item_quantized_codes_to_item_ids[sampled_code]
                    if len(item_ids_list) > 0:
                        break

            outfit_item_ids.append(item_ids_list[0])

        return outfit_item_ids

    def save_generated_outfit_new(self, name, sample_idx, generated_outfits):
        n_generated_outfits = len(generated_outfits)

        # Set up the figure
        generated_outfit_len = len(generated_outfits[0])
        fig, axs = plt.subplots(n_generated_outfits, generated_outfit_len, figsize=(16, 4))

        fig.suptitle(f"Sample#{sample_idx}", fontsize=16)

        # Display the generated outfits on bottom in order of ranking
        for i in range(n_generated_outfits):
            generated_outfit = generated_outfits[i]
            generated_outfit_len = len(generated_outfit)
            try:
                generated_outfit_item_ids = self.outfit_item_codes_to_outfit_item_ids(generated_outfit)
            except:
                plt.close()
                return
            print(f"generated_outfit_item_ids: {generated_outfit_item_ids}")
            for j in range(generated_outfit_len):
                generated_outfit_items_images = [self._load_image(item_id) for item_id in generated_outfit_item_ids]

                if n_generated_outfits > 1:
                    axs[i, j].imshow(generated_outfit_items_images[j])
                    axs[i, j].axis('off')  # hide axis
                else:
                    axs[j].imshow(generated_outfit_items_images[j])
                    axs[j].axis('off')  # hide axis

        
        generated_outfits_dir = os.path.join(self.model_artifacts_dir, "generated_outfits", str(name))
        if not os.path.exists(generated_outfits_dir):
            os.makedirs(generated_outfits_dir)

        img_path = os.path.join(generated_outfits_dir, f"sample#{str(sample_idx)}.png")
        print(f"Saving to {img_path}")
        plt.savefig(img_path)
        plt.close()
    
    def _construct_outfit_item_codes(self, outfit_id, max_outfit_len, add_masked_item_placeholder=False):
        # outfit_index = self.outfit_id_to_index[outfit_id]
        outfit_items_idxs = self.outfit_id_to_item_idxs[outfit_id]
        outfit_len = len(outfit_items_idxs)
        if add_masked_item_placeholder: outfit_len += 1

        outfit_items_idxs_tensor = torch.tensor(outfit_items_idxs, dtype=torch.int)
        outfit_items_codes_tensor = self.item_idx_to_quantized_code[outfit_items_idxs_tensor].clone()

        # BOS + N items start token + N_codes per N items  + EOS
        # sequence_length = 1 + outfit_len + (outfit_len * codebook_n_levels) + 1
        sequence_length = 1 + max_outfit_len + (max_outfit_len * self.codebook_n_levels) + 1
        # tokenized_outfit_sequence = torch.zeros(sequence_length, codebook_n_levels, dtype=torch.long)
        tokenized_outfit_sequence = torch.full((sequence_length,), self.pad_token_id, dtype=torch.long)
        outfit_item_seqmentation = torch.full((sequence_length,), 0, dtype=torch.long)

        # Set the start sequence token
        tokenized_outfit_sequence[0] = self.outfit_start_token_id
        # Insert the item start token and item tokens
        last_item_start_pos = -1
        for i in range(outfit_len):
            start_index = 1 + i * (self.codebook_n_levels + 1)  # Calculate the start index for each item's tokens
            if self.item_start_token_beginning:
                tokenized_outfit_sequence[start_index] = self.item_start_token_id  # Set the item_start_token_id
                # Copy the item's tuple of token Ids
                if add_masked_item_placeholder and i == outfit_len - 1:
                    masked_item_placedholder = torch.full((self.codebook_n_levels,), self.mask_token_id, dtype=torch.long)
                    tokenized_outfit_sequence[start_index + 1 : start_index + self.codebook_n_levels + 1] = masked_item_placedholder
                else:
                    tokenized_outfit_sequence[start_index + 1 : start_index + self.codebook_n_levels + 1] = outfit_items_codes_tensor[i]
                if i == outfit_len - 1:
                    last_item_start_pos = start_index
            else:
                if add_masked_item_placeholder and i == outfit_len - 1:
                    masked_item_placedholder = torch.full((self.codebook_n_levels,), self.mask_token_id, dtype=torch.long)
                    tokenized_outfit_sequence[start_index : start_index + self.codebook_n_levels] = masked_item_placedholder
                else:
                    tokenized_outfit_sequence[start_index : start_index + self.codebook_n_levels] = outfit_items_codes_tensor[i]
                tokenized_outfit_sequence[start_index + self.codebook_n_levels] = self.item_start_token_id  # Set the item_start_token_id
                if i == outfit_len - 1:
                    last_item_start_pos = start_index - 1
            outfit_item_seqmentation[start_index : start_index + self.codebook_n_levels + 1] = i + 1


        # Set the end_sequence_token_id after the last item
        end_position = 1 + outfit_len * (self.codebook_n_levels + 1)
        tokenized_outfit_sequence[end_position] = self.outfit_end_token_id

        return tokenized_outfit_sequence, outfit_item_seqmentation, last_item_start_pos
    
    def _tokenize_outfit(self, outfit_items_idxs):
        # print(f"outfit_items_idxs: {outfit_items_idxs}")

        outfit_items_idxs_tensor = torch.tensor(outfit_items_idxs, dtype=torch.int)

        outfit_items_codes_tensor = self.item_idx_to_quantized_code[outfit_items_idxs_tensor].clone()
        outfit_items_code_embs_tensor = self.item_idx_to_quantized_code_embs[outfit_items_idxs_tensor].clone()

        latent_dim = outfit_items_code_embs_tensor.shape[-1]
        # print(f"outfit_items_codes_tensor: {outfit_items_codes_tensor.shape}")
        # print(f"outfit_items_code_embs_tensor: {outfit_items_code_embs_tensor.shape}")

        outfit_items_codes_flattened = outfit_items_codes_tensor.flatten()
        outfit_items_code_embs_flattened = outfit_items_code_embs_tensor.view(-1, latent_dim)
        # print(f"outfit_items_codes_flattened: {outfit_items_codes_flattened.shape}")
        # print(f"outfit_items_code_embs_flattened: {outfit_items_code_embs_flattened.shape}")

        # Add special tokens:
        bos_token = torch.tensor([self.bos_token_id], dtype=torch.long)
        eos_token = torch.tensor([self.eos_token_id], dtype=torch.long)
        outfit_items_codes = torch.cat((bos_token, outfit_items_codes_flattened, eos_token))

        bos_token_emb = self.special_tokens_to_embs[self.bos_token_id].clone()
        eos_token_emb = self.special_tokens_to_embs[self.eos_token_id].clone()
        outfit_items_code_embs = torch.cat((bos_token_emb, outfit_items_code_embs_flattened, eos_token_emb))
        # print(f"outfit_items_codes: {outfit_items_codes.shape}")
        # print(f"outfit_items_code_embs: {outfit_items_code_embs.shape}")

        return outfit_items_codes, outfit_items_code_embs
    
    
    def _tokenize_outfit_old(self, outfit_items_idxs):
        outfit_len = len(outfit_items_idxs)
        # print(f"outfit_items_idxs: {outfit_items_idxs}")

        outfit_items_idxs_tensor = torch.tensor(outfit_items_idxs, dtype=torch.int)
        outfit_items_codes_tensor = self.item_idx_to_quantized_code[outfit_items_idxs_tensor].clone()
        # print(f"outfit_items_idxs_tensor: {outfit_items_idxs_tensor}")
        # print(f"outfit_items_codes_tensor: {outfit_items_codes_tensor}")
        # print(f"item_idx_to_quantized_code: {self.item_idx_to_quantized_code.shape}")

        # N items start token + N_codes per N items
        sequence_length = outfit_len * (self.codebook_n_levels + 1)
        # print(f"outfit_len: {outfit_len}")
        tokenized_outfit_sequence = torch.full((sequence_length,), self.pad_token_id, dtype=torch.long)

        for i in range(outfit_len):
            start_index = i * (self.codebook_n_levels + 1)  # Calculate the start index for each item's tokens
            if self.item_start_token_beginning:
                tokenized_outfit_sequence[start_index] = self.item_start_token_id  # Set the item_start_token_id
                # print(f"tokenized_outfit_sequence: {tokenized_outfit_sequence.shape}")
                # print(f"outfit_items_codes_tensor: {outfit_items_codes_tensor.shape}")
                tokenized_outfit_sequence[start_index + 1 : start_index + 1 + self.codebook_n_levels] = outfit_items_codes_tensor[i]
                if self.use_retrieval_token:
                    tokenized_outfit_sequence[start_index + self.codebook_n_levels] = self.retrieval_token_id
            else:
                tokenized_outfit_sequence[start_index : start_index + self.codebook_n_levels] = outfit_items_codes_tensor[i]
                tokenized_outfit_sequence[start_index + self.codebook_n_levels] = self.item_start_token_id  # Set the item_start_token_id
                if self.use_retrieval_token:
                    tokenized_outfit_sequence[start_index + self.codebook_n_levels - 1] = self.retrieval_token_id

        return tokenized_outfit_sequence
    
    
    def shuffle(self):
        self.split_set_outfit_ids = random.sample(self.split_set_outfit_ids, len(self.split_set_outfit_ids))

    


CATEGORY_INDEX_TO_PRIORITY = {
    45: 0,
    4: 1,
    174: 2,
    49: 3,
    8: 4,
    35: 5,
    148: 6,
    227: 7,
    68: 8,
    196: 9,
    222: 10,
    89: 11,
    173: 12,
    116: 13,
    119: 14,
    163: 15,
    63: 16,
    167: 17,
    142: 18,
    60: 19
}


ITEM_CLASSIFIERS_TEXT = [
    'dress',
    'blouse',
    'tank top',
    'crop top',
    'corset',
    'sweater',
    'cardigan',
    'coat',
    'jacket',
    'blazer',
    'skirt',
    'pants',
    'shorts',
    'jeans',
    'jumpsuit',
    'romper',
    'activewear',
    'outerwear',
    'sports bra',
    'leggings',
    'swimwear',
    'bikini',
    'dress shirt',
    'polo shirt',
    't-shirt',
    'shirt',
    'outerwear vest',
    'trousers',
    'suit',
    'sweatpants',
    'hoodie',
    'sweatshirt',
    'shoes',
    'footwear',
    'flats shoes',
    'heels',
    'boots',
    'sandals',
    'sneakers',
    'loafers',
    'dress shoes',
    'athletic shoes',
    'slippers',
    'bags', 
    'handbags',
    'backpack',
    'clutches',
    'pocketbook',
    'jewelry', 
    'necklace',
    'earrings',
    'bracelets',
    'rings',
    'sunglasses', 
    'glasses',
    'scarves', 
    'hats', 
    'beanies',
    'fedoras',
    'earmuffs',
    'headband',
    'headphones',
    'belts',
    'gloves',
    'ties',
    'socks',
    'watches',
    'digital watch',
    'umbrella',
    'clothing with text',
    'person', 
    'person face',
    'person head',
    'person hair',
    'group of people',
    'text only',
    'other',
    'furniture'
]

INVALID_ITEM_CLASSES = [
    'person', 
    'person face',
    'person head',
    'person hair',
    'group of people',
    'text only',
    'other',
    'furniture',
    'clothing with text',
    'earmuffs',
    'headband'
]