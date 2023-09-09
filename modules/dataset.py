import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

class DataModule:

    def __init__(self, dataset_type, base_dir):
        
        self.base_dir = base_dir
        self.image_cardinality = {}
        self.dataset = self.load_data(dataset_type)
        self.given_phrases = [sample['given_phrase'] for sample in self.dataset]
        

    def load_image(self, image_path):
        return Image.open(image_path).convert('RGB')

    def load_data(self, dataset_type):
        if dataset_type == 'train':
            TRAIN_DATA_FILE_PATH = self.base_dir + '/train_v1/'
            IMAGES_DIR = TRAIN_DATA_FILE_PATH + '/train_images_v1/' 
            DATA_FILE = TRAIN_DATA_FILE_PATH + 'train.data.v1.txt' 
            GOLD_DATA_FILE = TRAIN_DATA_FILE_PATH + 'train.gold.v1.txt'
        elif dataset_type == 'test':
            TRAIN_DATA_FILE_PATH = self.base_dir + '/test/'
            IMAGES_DIR = TRAIN_DATA_FILE_PATH + '/test_images/'
            DATA_FILE = TRAIN_DATA_FILE_PATH + 'en.test.data.v1.1.txt'
            GOLD_DATA_FILE = TRAIN_DATA_FILE_PATH + 'en.test.gold.v1.1.txt'

        data = []
        self.phrase_idx_mapping = {}
        with open(DATA_FILE) as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line_splitted = line.split()
                images_index = [i for i in range(len(line_splitted)) if ('.jpg' in line_splitted[i]) or ('.png' in line_splitted[i]) or ('.PNG' in line_splitted[i]) or ('.jpeg' in line_splitted[i]) or ('.JPG' in line_splitted[i])][0]
                data.append({
                    'word': line_splitted[0], 
                    'given_phrase': " ".join(line_splitted[1:images_index]),
                    'images_paths': [IMAGES_DIR + x for x in line_splitted[images_index:]],
                    'images_names': line_splitted[images_index:]
                })
                assert len(data[-1]['images_paths']) == 10
                self.phrase_idx_mapping[data[-1]['given_phrase']] = idx

                # Update image cardinality
                for img_name in data[-1]['images_names']:
                    if img_name in self.image_cardinality:
                        self.image_cardinality[img_name] += 1
                    else:
                        self.image_cardinality[img_name] = 1

        with open(GOLD_DATA_FILE) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                data[i]['gold_image_path'] = IMAGES_DIR + line.split('\n')[0]
                data[i]['gold_image_index'] = data[i]['images_paths'].index(data[i]['gold_image_path'])
        return data


    def __getitem__(self, item):
        
        if isinstance(item, str):
            ret_item = self.dataset[self.phrase_idx_mapping[item]]
            ret_item['images'] = [self.load_image(img_path) for img_path in ret_item['images_paths']]
        
        else:
            ret_item = self.dataset[item]
            ret_item['images'] = [self.load_image(img_path) for img_path in ret_item['images_paths']]
        return ret_item        

    def __len__(self):
        return len(self.dataset)


class Dataset:

    def __init__(self, base_dir=None):

        self.base_dir = base_dir if base_dir is not None else os.getenv("DATASET_PATH")
        if self.base_dir is None:
            raise ValueError("DATASET_PATH is needed")
    
    def train_dataloader(self):
        return DataModule(dataset_type='train', base_dir=self.base_dir)
    
    def test_dataloader(self):
        return DataModule(dataset_type='test', base_dir=self.base_dir)
