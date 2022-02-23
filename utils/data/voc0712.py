"""VOC Dataset Classes
Original author: Yonghye Kwon
https://github.com/developer0hye
"""
import cv2
import numpy as np

import os
import xml.etree.ElementTree as ET

CLASSES = (
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor")

class VOCDetection(object):
    
    def __init__(self, 
                 root, 
                 image_sets=[("2007", "trainval"), ("2012", "trainval")],
                 keep_difficult=False):
        """VOC Detection Dataset Object
        input is image, target is annotation
        Arguments:
            root (string): filepath to VOCdevkit folder.
            image_set (string): imageset to use (eg. "train", "val", "trainval", "test")
            keep_difficult (boolean): 
        """
        self.root = root
        self.imgs_path = []
        self.labels_path = []
        self.labels = []
        
        self.keep_difficult = keep_difficult
        
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root, "VOC" + year)
            with open(os.path.join(rootpath, "ImageSets", "Main", name + ".txt")) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    
                    img_path = os.path.join(rootpath, "JPEGImages", line + ".jpg")
                    label_path = os.path.join(rootpath, "Annotations", line + ".xml")
                    label = self.read_xml(label_path)
                    
                    self.imgs_path.append(img_path)
                    self.labels_path.append(label_path)
                    self.labels.append(label)
        
        assert len(self.imgs_path) == len(self.labels_path)
        assert len(self.labels_path) == len(self.labels)
        
    def read_xml(self, label_path):
        label = []
        
        tree = ET.parse(label_path)
        root = tree.getroot()
        
        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)
        
        objs = root.findall("object")
        for obj in objs:
            difficult = int(obj.find("difficult").text)
            
            if difficult == 1 and self.keep_difficult == False:
                continue

            class_name = obj.find("name").text.lower().strip()
            
            if not class_name in CLASSES:
                continue
            
            class_idx = CLASSES.index(class_name)

            bbox = obj.find("bndbox")
            
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            cx = ((xmax + xmin)/2.)/img_w
            cy = ((ymax + ymin)/2.)/img_h
            w = (xmax - xmin)/img_w
            h = (ymax - ymin)/img_h
            
            label.append([class_idx, cx, cy, w, h])
        
        label = np.array(label).reshape(-1, 5)
        label[:, 1:] = np.clip(label[:, 1:], a_min=0., a_max=1.)
        return label
    
    def __getitem__(self, idx):
        img = cv2.imread(self.imgs_path[idx], cv2.IMREAD_COLOR)
        label = self.labels[idx].copy()
        return img, label, self.imgs_path[idx]
    
    def __len__(self): 
        return len(self.imgs_path)
    