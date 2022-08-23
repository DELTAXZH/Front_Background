"""
self supervise dataset AI-inferance    Script  ver： Aug 21th 23:50

"""

import os
import cv2
import torch
import numpy as np

def find_all_files(root, suffix=None):
    """
    Return a list of file paths ended with specific suffix
    """
    res = []
    if type(suffix) is tuple or type(suffix) is list:
        for root, _, files in os.walk(root):
            for f in files:
                if suffix is not None:
                    status = 0
                    for i in suffix:
                        if not f.endswith(i):
                            pass
                        else:
                            status = 1
                            break
                    if status == 0:
                        continue
                res.append(os.path.join(root, f))
        return res

    elif type(suffix) is str or suffix is None:
        for root, _, files in os.walk(root):
            for f in files:
                if suffix is not None and not f.endswith(suffix):
                    continue
                res.append(os.path.join(root, f))
        return res

    else:
        print('type of suffix is not legal :', type(suffix))
        return -1


class Front_Background_Dataset(torch.utils.data.Dataset):
    def __init__(self, input_root, data_transforms=None, edge_size=384, suffix='.jpg'):

        super().__init__()

        self.data_root = os.path.join(input_root, 'data')

        # get files
        self.input_ids = sorted(find_all_files(self.data_root, suffix=suffix))

        # get data augmentation and transform
        if data_transforms is not None:
            self.transform = data_transforms
        else:
            self.transform = transforms.Compose([transforms.Resize(edge_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # get data path
        imageName = self.input_ids[idx]
        # get image id
        imageID = os.path.split(imageName)[-1].split('.')[0]

        # get data
        # CV2 0-255 hwc，in totensor step it will be transformed to chw.  ps:PIL(0-1 hwc)
        image = np.array(cv2.imread(imageName), dtype=np.float32)

        image = self.transform(image)

        return image, imageID


def inferance(model, dataloader, record_dir, result_csv_name='inferance.csv', device='cuda'):
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    model.eval()
    print('Inferance')
    print('-' * 10)

    check_idx = 0

    with open(os.path.join(record_dir, result_csv_name), 'w') as f_log:
        # Iterate over data.
        for images, imageIDs in dataloader:
            images = images.to(device)

            # forward
            outputs = model(images)
            confidence, preds = torch.max(outputs, 1)

            pred_labels = preds.cpu().numpy()

            for output_idx in range(len(pred_labels)):
                f_log.write(str(imageIDs[output_idx]) + ', ' + str(pred_labels[output_idx]) + ', \n')
                check_idx += 1

        f_log.close()
        print(str(check_idx) + ' samples are all recorded')


# input root: /Users/xuzhaohan/Desktop/XZH/PB/data/warwick_CLS