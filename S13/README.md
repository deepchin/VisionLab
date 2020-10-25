## yolo object detection using opencv  
## Yolo object detection using CNN model

### Authors

* **Deepak Hazarika** 
* **Parinita Bora**
* **Gurudatta**

10/25/2020 12:17:59 PM  

----------

# Object detection using opencv

Use the code from 

https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/

to detect object in an image.The following file is used to detect object in the image

[yolo object detection](https://github.com/deepchin/VisionLab/blob/master/S13/yolo_object_detection/yolo_object_detection.py "yolo object detection")

An example annotated image is shown below

![Annotated image](yolo_object_detection/Deepak_3_annotated.jpg)

# Object detection using CNN

A sample dataset of 3518 images were annotated with 4 classes

1. hardhat
2. mask
3. vest
4. boots  

The dataset was split into 

1. train set - 3200 images
2. test set - 318 images

The following repository was used to train and test the dataset.

[YoloV3 repo](https://github.com/theschoolofai/YoloV3 "YoloV3 repo")

The trained model was tested on random images to detect the four classes mentioned above.A video link of the out is given below.

[Object detection video](https://github.com/deepchin/VisionLab/blob/master/S13/S13_Yolo_video.mp4 "Object detection video ")

[You tube Object detection video](https://youtu.be/KY3u723XVws "Object detection video")


## Training log

Namespace(accumulate=4, adam=False, batch_size=10, bucket='', cache_images=True, cfg='cfg/yolov3-custom.cfg', data='data/customdata/custom.data', device='', epochs=20, evolve=False, img_size=[512], multi_scale=False, name='', nosave=True, notest=False, rect=False, resume=False, single_cls=False, weights='weights/yolov3-spp-ultralytics.pt')
Using CUDA device0 _CudaDeviceProperties(name='Tesla P100-PCIE-16GB', total_memory=16280MB)

2020-10-24 14:34:02.663104: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/
Model Summary: 225 layers, 6.25895e+07 parameters, 6.25895e+07 gradients
Caching labels (3031 found, 131 missing, 38 empty, 0 duplicate, for 3200 images): 100% 3200/3200 [00:00<00:00, 8060.75it/s]
Caching images (1.8GB): 100% 3200/3200 [00:20<00:00, 154.57it/s]
Reading image shapes: 100% 318/318 [00:00<00:00, 4554.22it/s]
Caching labels (297 found, 14 missing, 7 empty, 0 duplicate, for 318 images): 100% 318/318 [00:00<00:00, 7898.04it/s]
Caching images (0.1GB): 100% 318/318 [00:02<00:00, 106.84it/s]
Image sizes 512 - 512 train, 512 test
Using 2 dataloader workers
Starting training for 20 epochs...

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
  0% 0/320 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/torch/cuda/memory.py:346: FutureWarning: torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved
  FutureWarning)
      0/19     7.37G      8.26      6.62      3.48      18.4        54       512:   0% 1/320 [00:03<17:24,  3.27s/it]/usr/local/lib/python3.6/dist-packages/torch/cuda/memory.py:346: FutureWarning: torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved
  FutureWarning)
      0/19     7.71G      4.93      3.42      2.25      10.6        74       512: 100% 320/320 [03:35<00:00,  1.48it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1:   0% 0/32 [00:00<?, ?it/s]/content/YoloV3-EVA5_S13/utils/utils.py:530: UserWarning: This overload of nonzero is deprecated:
	nonzero()
Consider using one of the following signatures instead:
	nonzero(*, bool as_tuple) (Triggered internally at  /pytorch/torch/csrc/utils/python_arg_parser.cpp:766.)
  i, j = (x[:, 5:] > conf_thres).nonzero().t()
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:08<00:00,  3.77it/s]
                 all       318  1.53e+03     0.284     0.495     0.311     0.356

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
  0% 0/320 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/torch/cuda/memory.py:346: FutureWarning: torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved
  FutureWarning)
      1/19     7.72G      3.51      1.89      1.04      6.44        62       512: 100% 320/320 [03:33<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  4.58it/s]
                 all       318  1.53e+03     0.448     0.554     0.422     0.488

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      2/19     7.72G      3.09      1.71     0.893      5.69        90       512: 100% 320/320 [03:33<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  4.68it/s]
                 all       318  1.53e+03     0.445      0.65     0.518     0.524

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
  0% 0/320 [00:00<?, ?it/s]
Model Bias Summary:    layer        regression        objectness    classification
                          89      -0.15+/-0.20      -5.28+/-0.52      -1.55+/-0.32 
                         101       0.03+/-0.23      -5.24+/-0.29      -1.53+/-0.19 
                         113       0.15+/-0.21      -6.00+/-0.24      -1.54+/-0.27 
      3/19     7.72G       2.9      1.62     0.838      5.36        56       512: 100% 320/320 [03:33<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  4.76it/s]
                 all       318  1.53e+03     0.497     0.666     0.538     0.563

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      4/19     7.72G       2.7      1.51      0.77      4.98        88       512: 100% 320/320 [03:33<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  4.84it/s]
                 all       318  1.53e+03     0.504     0.671     0.533     0.569

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      5/19     7.72G      2.61      1.41     0.732      4.76        46       512: 100% 320/320 [03:33<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  4.83it/s]
                 all       318  1.53e+03     0.531     0.637     0.552     0.579

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      6/19     7.72G       2.5      1.39     0.658      4.55        74       512: 100% 320/320 [03:32<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  4.86it/s]
                 all       318  1.53e+03     0.565     0.651      0.55     0.601

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      7/19     7.72G      2.45      1.35     0.617      4.41        53       512: 100% 320/320 [03:33<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  4.86it/s]
                 all       318  1.53e+03     0.545     0.691     0.571     0.608

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      8/19     7.72G      2.38       1.3     0.549      4.23        58       512: 100% 320/320 [03:32<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  4.84it/s]
                 all       318  1.53e+03     0.557      0.67     0.562     0.607

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      9/19     7.72G      2.32      1.28     0.487      4.08        77       512: 100% 320/320 [03:32<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  4.93it/s]
                 all       318  1.53e+03     0.518     0.675     0.565     0.584

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     10/19     7.72G      2.25      1.25      0.47      3.97        67       512: 100% 320/320 [03:32<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  4.90it/s]
                 all       318  1.53e+03     0.535     0.687     0.554     0.601

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     11/19     7.72G      2.22      1.25     0.392      3.86        74       512: 100% 320/320 [03:32<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  4.93it/s]
                 all       318  1.53e+03     0.522     0.698     0.561     0.596

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     12/19     7.72G      2.12      1.21     0.354      3.68        37       512: 100% 320/320 [03:32<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  4.98it/s]
                 all       318  1.53e+03      0.56     0.668     0.552     0.609

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     13/19     7.72G      2.07      1.17     0.305      3.54        63       512: 100% 320/320 [03:32<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  4.93it/s]
                 all       318  1.53e+03     0.519     0.692     0.562     0.592

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     14/19     7.72G      2.03      1.16     0.283      3.47        50       512: 100% 320/320 [03:32<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  5.00it/s]
                 all       318  1.53e+03     0.547     0.687     0.561     0.608

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     15/19     7.72G      1.98      1.13     0.264      3.38        51       512: 100% 320/320 [03:32<00:00,  1.51it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  4.99it/s]
                 all       318  1.53e+03     0.544      0.69     0.561     0.607

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     16/19     7.72G      1.93       1.1     0.234      3.26        54       512: 100% 320/320 [03:32<00:00,  1.51it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  5.01it/s]
                 all       318  1.53e+03     0.532     0.692     0.556     0.601

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     17/19     7.72G      1.89      1.08     0.196      3.17        67       512: 100% 320/320 [03:32<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  5.03it/s]
                 all       318  1.53e+03     0.539      0.69     0.559     0.605

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     18/19     7.72G      1.88      1.08     0.178      3.14        65       512: 100% 320/320 [03:32<00:00,  1.50it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  5.03it/s]
                 all       318  1.53e+03     0.545     0.682     0.563     0.605

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     19/19     7.72G      1.86      1.05     0.193      3.11        70       512: 100% 320/320 [03:32<00:00,  1.51it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:06<00:00,  5.02it/s]
                 all       318  1.53e+03     0.544     0.689     0.556     0.607
20 epochs completed in 1.221 hours.