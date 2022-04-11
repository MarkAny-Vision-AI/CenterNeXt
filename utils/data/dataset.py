from . import voc0712
from . import coco
from . import transforms

import cv2
import numpy as np
import torch 
from torch.utils.data import Dataset

class DetectionDataset(Dataset):  # for training/testing
    def __init__(self,
                 option,
                 split,
                 apply_augmentation=False):
        
        assert split == "train" or split == "val"
        
        if option["DATASET"]["NAME"] == "VOC0712":
            self.dataset = voc0712.VOCDetection(option["DATASET"]["ROOT"], 
                                                [("2007", "trainval"), ("2012", "trainval")] if split == "train" else [("2007", "test")], 
                                                keep_difficult=False)
        elif option["DATASET"]["NAME"] == "COCO":
            self.dataset = coco.COCODetection(option["DATASET"]["ROOT"],
                                              "train2017" if split == "train" else "val2017")
        
        self.num_classes = len(option["MODEL"]["CLASSES"])
        
        self.img_w = option["MODEL"]["INPUT_SIZE"]["WIDTH"]
        self.img_h = option["MODEL"]["INPUT_SIZE"]["HEIGHT"]
        self.stride = option["MODEL"]["STRIDE"]
        
        self.heatmap_w = self.img_w // self.stride
        self.heatmap_h = self.img_h // self.stride
        
        self.keep_ratio = True
        self.apply_augmentation = apply_augmentation
        self.use_mosaic = option["MODEL"]["USE_MOSAIC"]
        self.use_mixup = option["MODEL"]["USE_MIXUP"]
        
    def __getitem__(self, idx):
        img, label, img_path = self.dataset[idx]
        org_img_shape = [img.shape[1], img.shape[0]]
        padded_ltrb = [0, 0, 0, 0]
        
        bboxes_class = label[:, 0].reshape(-1, 1)
        bboxes_cxcywh = label[:, 1:].reshape(-1, 4)

        #resize image
        if self.keep_ratio:
            img, bboxes_cxcywh, org_img_shape, padded_ltrb  = transforms.aspect_ratio_preserved_resize(img,
                                                                                                       dsize=(self.img_w, self.img_h),
                                                                                                       bboxes_cxcywh=bboxes_cxcywh)
        else:        
            img = cv2.resize(img, dsize=(self.img_w, self.img_h))
        
        #augmentation
        if self.apply_augmentation:
            img, bboxes_cxcywh, bboxes_class = transforms.random_crop(img, bboxes_cxcywh, bboxes_class, p=1.0)
            
            if self.use_mosaic:
                img, bboxes_cxcywh, bboxes_class = transforms.mosaic(img, bboxes_cxcywh, bboxes_class, self.dataset, self.keep_ratio, p=0.5)

            if self.use_mixup:  
                img, bboxes_cxcywh, bboxes_class = transforms.mixup(img, bboxes_cxcywh, bboxes_class, self.dataset, self.keep_ratio, use_mosaic=self.use_mosaic, p=0.5, mosaic_p=0.5)
    
            img, bboxes_cxcywh = transforms.horizontal_flip(img, bboxes_cxcywh, p=0.5)
            img, bboxes_cxcywh = transforms.random_translation(img, bboxes_cxcywh, p=1.0)
            img, bboxes_cxcywh = transforms.random_scale(img, bboxes_cxcywh, p=1.0, bbox_wmin_thr=16, bbox_hmin_thr=16)
            img = transforms.augment_hsv(img)

        #numpy(=opencv)img 2 pytorch tensor        
        img = img[..., ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.tensor(img, dtype=torch.float32)/255.
        
        label = np.concatenate([bboxes_class, bboxes_cxcywh], axis=1)
        annotations = label.copy()
        
        label[:, 1:] = np.clip(label[:, 1:], a_min=0., a_max=1.)

        label[:, [1, 3]] *= self.heatmap_w
        label[:, [2, 4]] *= self.heatmap_h
        
        label = label[ (label[:, 3] * self.stride >= 3) & (label[:, 4] * self.stride >= 3) ] # size filtering
        
        bboxes_regression = np.zeros(shape=(4, self.heatmap_h, self.heatmap_w), dtype=np.float32)
        classes_gaussian_heatmap = np.zeros(shape=(self.num_classes, self.heatmap_h, self.heatmap_w), dtype=np.float32)
        foreground = np.zeros(shape=(self.heatmap_h, self.heatmap_w), dtype=np.float32)
        
        bboxes_icx = label[:, 1].astype(np.int)
        bboxes_icy = label[:, 2].astype(np.int)
        
        foreground[bboxes_icy, bboxes_icx] = 1
        bboxes_regression[0, bboxes_icy, bboxes_icx] = label[:, 1] - bboxes_icx
        bboxes_regression[1, bboxes_icy, bboxes_icx] = label[:, 2] - bboxes_icy
        bboxes_regression[2, bboxes_icy, bboxes_icx] = label[:, 3]
        bboxes_regression[3, bboxes_icy, bboxes_icx] = label[:, 4]
        
        for bbox in label:
            bbox_class = int(bbox[0])
            bbox_fcx, bbox_fcy, bbox_w, bbox_h = bbox[1:]
            bbox_icx, bbox_icy = int(bbox_fcx), int(bbox_fcy)            
            classes_gaussian_heatmap[bbox_class] = transforms.scatter_gaussian_kernel(classes_gaussian_heatmap[bbox_class], bbox_icx, bbox_icy, bbox_w.item(), bbox_h.item())

        annotations = torch.tensor(annotations)
        bboxes_regression = torch.tensor(bboxes_regression)
        classes_gaussian_heatmap = torch.tensor(classes_gaussian_heatmap)
        foreground = torch.tensor(foreground)

        data = {}
        data["img"] = img
        data["label"] = {"annotations": annotations,"bboxes_regression": bboxes_regression, "classes_gaussian_heatmap": classes_gaussian_heatmap, "foreground": foreground}
        data["img_path"] = img_path
        data["org_img_shape"] = org_img_shape
        data["padded_ltrb"] = padded_ltrb
        
        return data
    
    def __len__(self):
        return len(self.dataset)
    
def collate_fn(batch_data):
    batch_img = []
    batch_annotations = []
    batch_bboxes_regression = []
    batch_classes_gaussian_heatmap = []
    batch_foreground = []
    batch_img_path = []
    batch_org_img_shape = []
    batch_padded_ltrb = []

    for data in batch_data:
        batch_img.append(data["img"])
        batch_annotations.append(data["label"]["annotations"])
        batch_bboxes_regression.append(data["label"]["bboxes_regression"])
        batch_classes_gaussian_heatmap.append(data["label"]["classes_gaussian_heatmap"])
        batch_foreground.append(data["label"]["foreground"])
        batch_img_path.append(data["img_path"])
        batch_org_img_shape.append(data["org_img_shape"])
        batch_padded_ltrb.append(data["padded_ltrb"])

    batch_img = torch.stack(batch_img, 0)
    batch_bboxes_regression = torch.stack(batch_bboxes_regression, 0)
    batch_classes_gaussian_heatmap = torch.stack(batch_classes_gaussian_heatmap, 0)
    batch_foreground = torch.stack(batch_foreground, 0)
    
    batch_data = {}
    batch_data["img"] = batch_img
    batch_data["label"] = {"annotations": batch_annotations, "bboxes_regression": batch_bboxes_regression, "classes_gaussian_heatmap": batch_classes_gaussian_heatmap, "foreground": batch_foreground}
    batch_data["img_path"] = batch_img_path
    batch_data["org_img_shape"] = batch_org_img_shape
    batch_data["padded_ltrb"] = batch_padded_ltrb

    return batch_data