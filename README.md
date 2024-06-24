# Self-Supervised Learning for Image Segmentation

This repository contains an implementation of *[Simple Masked Image Modelling (SimMIM)](https://arxiv.org/abs/2111.09886)*, a self-supervised learning framework for pre-training vision transformers for downstream tasks such as classification or segmentation. A [VisionTransformer](https://arxiv.org/abs/2010.11929) model is pre-trained on the [ImageNet-1K](https://www.image-net.org/) dataset and fine-tuned on the [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset for segmentation.

Example reconstructions from the pre-trained encoder (original image, masked image and reconstruction from left to right for each example):
<p float="left">
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/9f101494-8acc-4fdc-b4b2-2056b7605d3e" width="16%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/e3956211-3282-4a62-96d0-62a1dde1ae88" width="16%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/e8de8c7f-7913-4328-b5b3-db0c15aa323a" width="16%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/c56fe53a-53e6-49d1-acfd-8c718e24bddf" width="16%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/b2f3a604-aa03-4d27-9088-640c2107f64b" width="16%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/601a49fa-3548-4c2b-a881-04eb2fa57ccd" width="16%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/5ab3937c-f7b2-4ffe-9722-5b230548f6cd" width="16%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/ae1b651c-e3ef-4e79-bd2e-2e64070fbf08" width="16%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/d2ad7ec4-a961-409d-9b3e-5e94f6e0a371" width="16%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/9c5b6c03-6e4f-4ce4-b508-18baafa1b924" width="16%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/ad162811-80d6-460b-99f0-fdc82342e61f" width="16%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/91e21806-938e-4436-a45e-7590011bc190" width="16%" />
</p>

Example segmentation predictions from the fine-tuned model (image, ground-truth segmentation map and predicted map from left to right for each example):
<p float="left">
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/0f8a6523-fd4a-4251-8ad0-c9c42169baaa" width="49.5%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/858500d7-047d-4192-8f17-f57d237ca5da" width="49.5%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/501d47a9-0a60-450d-94cd-9d0978f2fb29" width="49.5%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/3e396e32-c48c-4a26-92a4-a10bbb827654" width="49.5%" />
</p>


## Requirements
To install requirements, run the command:

    pip install -r requirements.txt

## Self-Supervised Pre-Training with SimMIM on ImageNet-1K

1. Navigate to the project's `/data/` folder and download ImageNet-1K by either running these commands below in a bash command line, or manually using the links to these 3 files ([devkit](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz), [validation](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar), [train](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar)):
   
        cd data
        wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate
        wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
        wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
    Note that this may take upwards of **4 hours**, depending on your connection. You may choose to download the validation set and the devkit files only and train on this smaller subset, in which case use the `--val_set` flag in the next step.
   
2. Run the following to start pre-training (with the default settings used in the report, on a smaller subset of data):

        python main_pretrain.py \
            --config vit_4M_pretrain \
            --train_size 10000 \

   This will first extract the compressed ImageNet files, then start printing training statistics and save the model weights as a `.pth` file every epoch. Use the flag `--run_plots` to save reconstructions during training, and the `--val_set` flag to use the smaller (validation) set only, for quicker testing. Change the train size between 45k, 100k and 200k to reproduce results from our report.
   
## Fine-Tuning on Oxford-IIIT Pets
1. With the pre-trained encoder weights in the `/weights/` folder, run this command for fine-tuning, which will download the Oxford-IIIT Pets dataset and start training initialised with the weights given:

        python main_finetune.py \
            --config vit_4M_finetune \
            --train_size 6000 \
            --test_size 1000 \
            --weights weights/encoder_vit_4M_pretrain_200K.pth
   
   Loss is printed every epoch while test set pixel accuracy and mean IoU is calculated after training is complete.
   Segmentation predictions will be saved under `/figures/`. Change the train size to reproduce results from our report.
2. To run a baseline model with **no pre-training**, omit the `--weights` argument, i.e. use the following command:

        python main_finetune.py \
            --config vit_4M_finetune \
            --train_size 6000 \
            --test_size 1000 \

## Intermediate-Task Fine-Tuning on Intel Image Classification Dataset
1. First, the Intel Image Classification dataset needs to be downloaded. From the project's root directory, run:
        
        cd data
        wget https://huggingface.co/datasets/miladfa7/Intel-Image-Classification/resolve/main/archive.zip?download=true
        
2. With the pre-trained encoder weights in the `/weights/` folder, run the following command to perform intermediate fine-tuning on this dataset, followed by segmentation fine-tuning on Oxford-IIIT Pets:
    
            python main_finetune.py \
            --config vit_4M_finetune \
            --train_size 6000 \
            --test_size 1000 \
            --weights weights/encoder_vit_4M_pretrain_200K.pth \
            --int_finetune

## Evaluation
To plot reconstructions from pre-trained models on ImageNet validation set (download above):

        python evaluation.py \
            --config vit_4M_pretrain \
            --weights weights/mim_vit_4M_pretrain_200K.pth \

To evaluate a fine-tuned segmentation model on the Oxford-IIIt Pets test set, use a command like the following, replacing the weights with those saved after fine-tuning (see above):
    
        python evaluation.py \
            --config vit_4M_finetune \
            --weights weights/vit_4M_finetune_data_250.pth \
            --test_size 1000 \
            --train_size 250
