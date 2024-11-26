# MagicFixup
This is the repo for the paper [Magic Fixup: Streamlining Photo Editing by Watching Dynamic Videos](https://magic-fixup.github.io)

**NEW Released the User Interface!**

## Installation
We provide an `environment.yaml` file to assist with installation. All what you need for setup is to run the following script
```
conda env create -f environment.yaml -v
```
and this will create a conda environment that you can activate using `conda activate MagicFixup`

## Inference

#### Downloading Magic Fixup checkpoint
You can download the model trained on the Moments in Time dataset using this [Google Drive Link](https://drive.google.com/file/d/1zOcDcJzCijbGr9I9adC0Cv6yzW60U9TQ/view?usp=share_link) or from [HuggingFace] (https://huggingface.co/HadiZayer/MagicFixup)


### Inference script
The inference scripts is `run_magicfu.py`. It takes the path of the reference image (the original image), and the edited image. Note that it assumes that the alpha channel is set appropriately in the edited image PNG, as we use the alpha channel to set the disocclusion mask. You can run the inference script with

```
python run_magicfu.py --checkpoint <Magic Fixup checkpoint> --reference <path to original image> --edit <path to png user edit>
```

### gradio demo
We have a gradio demo that allows you to test out your inputs with a friendly user interface. Simply start the demo with
```
python magicfu_gradio.py  --checkpoint <Magic Fixup checkpoint>
```


## Training your own model
To train your own model, first you need to process a video dataset, train the model using the processed pairs from your videos. In our model, we used the Momnets in Time dataset to train the weights we provided above.

#### Pretrained SD1.4 diffusion model
We start training from the official SD1.4 model (with the first layer modified to take our 9 channel input). You can either download the official SD1.4 model and modify the first layer using `scripts/modify_checkpoints.py` and place it under `pretrained_models` folder.

### Data Processing
The data processing code can be found under the `data_processing` folder. You can simply put all the videos in a directory, and pass the directory as the folder name in `data_processing/moments_processing.py`. If your videos are long (~ex more than 5 seconds and contain cut scenes), then you would want to use pyscenedetect to detect cut scenes and split the videos accordingly.
For data processing, you also need to download the checkpoint for SegmentAnything, and install soft-splatting. You can setup softmax-splatting and SAM, by following 
```
cd data_processing
git clone https://github.com/sniklaus/softmax-splatting.git
pip install segment_anything
cd sam_model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
For softmax-splatting to run, you need to install `pip install cupy` (or you might need to use `pip install cupy-cuda11x` or `pip install cupy-cuda12x` depending on your cuda version, and load the appropriate cuda module)

Then run `python moments_processing.py` to start processing frames from the provided examples video (included under `data_processing/example_videos`). For the version provided, we used the [Moments in Time Dataset](http://moments.csail.mit.edu)

### Running the training script
Make sure that you have downloaded the pretrained SD1.4 model above.  Once you download the training dataset and pretrained model, you can simply start training the model with 
```
./train.sh
```
The training code is in `main.py`, and relies mainly on pytorch_lightning in training.

Note that you need to modify the train and val paths in the chosen config file to the location where you have the processed data.

Note: we use Deepspeed to lower the memory requirements, so the saved model weights will be sharded. The script to reconstruct the model weights will be created in the checkpoint directory with name `zero_to_fp32.py`. One bug in the file is that it wouldn't recognize files with deepspeed1 (which is the one we use), so simply find and replace the string `== 2` with the string `<= 2` and it will work.

### Saving the Full Model Weights
To save storage requirements, we only checkpoint the learnable parameters in training (i.e. the frozen autoencoder params are not saved). To create a checkpoint that contains all the parameters, you can combine the frozen pretrained weights and learned parameters by running

```
python combine_model_params.py --pretrained_sd <path to pretrained SD1.4 with modified first layer> --learned_params <path to combined checkpoint learned> --save_path <path to save the >
```

## Editing UI
To help making your edits easier, we have released the our segmenting based UI. See the [UI folder](https://github.com/adobe-research/MagicFixup/tree/main/UI) for instructions on how to use it and set it up.

## Bibtex
if you find our work useful, please consider citing it in your work

```
    @misc{alzayer2024magicfixup,
      title={Magic Fixup: Streamlining Photo Editing by Watching Dynamic Videos}, 
      author={Hadi Alzayer and Zhihao Xia and Xuaner Zhang and Eli Shechtman and Jia-Bin Huang and Michael Gharbi},
      year={2024},
      eprint={2403.13044},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.13044}, 
    }
```

##### Acknowledgement
The diffusion code was built on top of the codebase adapted in [PaintByExample](https://github.com/Fantasy-Studio/Paint-by-Example)