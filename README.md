# Audio-visual Deepfake Detection With Local Temporal Inconsistencies
This repository contains the implementation of Audio Visual Deepfake Detection method proposed in the paper:  
  
Marcella Astrid, Enjie Ghorbel, and Djamila Aouada, Audio-visual Deepfake Detection With Local Temporal Inconsistencies, ICASSP 2025.  
<!---Links: [[PDF]](https://arxiv.org/pdf/2501.08137)-->

# Dependencies
Create conda environment with package inside the package list `conda create -n myenv --file package-list.txt`
  
# Prepare data (DFDC)
1) Download DFDC dataset from [here](https://www.kaggle.com/c/deepfake-detection-challenge/data). 
  
2) Store the train and test videos as follows:  

   ```
   train/real/{videoname}.mp4  
   train/fake/{videoname}.mp4  
   test/real/{videoname}.mp4  
   test/fake/{videoname}.mp4
   ```
  
   If you wish to use the same videos as used by the paper authors, please refer to `train_fake_list.txt`,  `train_real_list.txt`,  `test_fake_list.txt`, and `test_real_list.txt`. 
  
3) Once the videos have been placed at the above mentioned paths, run `python pre-process.py --out_dir train` and `python pre-process.py --out_dir test` for pre-processing the videos.  
  
4) After the above step, you can delete `pyavi`, `pywork`, `pyframes` and `pycrop` directories under `train` and `test` folders. (Do not delete `pytmp` folder please!)  
  
5) Collect video paths in csv files by running `python write_csv.py --out_dir . ` command. Also can create the small set version (e.g., used in Table 3) with `python write_csv.py --out_dir . --small`  

# Prepare data (FakeAVCeleb)
1) Download FakeAVCeleb dataset from [here](https://github.com/DASH-Lab/FakeAVCeleb/blob/main/dataset/README.md)
2) Run `python preprocess_FakeAVCeleb_to_DFDCformat.py`. See `fakeavceleb_test_fake.txt` and `fakeavceleb_test_real.txt` for list of videos we are using.
3) Use instructions in Prepare data (DFDC). Adjust the `--out_dir` in step 3 respectively. Also add `--dont_crop_face` option in step 3. Use `write_csv_fakeavceleb.py` for step 5

# Training
```
python my_train.py --out_dir . --epochs 50 --num_workers 7 --with_att --using_pseudo_fake --temporal_size 7 
```
Remove `--with_att` for model without attention.

Remove `--using_pseudo_fake` for training without pseudo-fake.

Change the 7 in `--temporal_size 7` to change the temporal size. (Table 2 ablation)

More options, see my_train.py

Final model weight file: [drive](https://drive.google.com/drive/folders/1ahHa749xcir7wP10Uu3TWcDWMVtAWN6P?usp=sharing)

```
# Testing

python my_train.py --out_dir . --test log_tmp/v6_mydf-tems7-natt1.0-fingtemporal-upf-PF0,1,2_A2_afl2,-1_vfl2,-1_224_r18_bs8_lr0.0001/model/model_best_epoch50.pth.tar --with_att --temporal_size 7
```
Change the path of model file accordingly in the --test argument.  
  
For computing AUC score, run `python my_test.py --folder test_results/v6_mydf-tems7-natt1.0-fingtemporal-upf-PF0,1,2_A2_afl2,-1_vfl2,-1_224_r18_bs8_lr0.0001/model_best_epoch50/imbalance` after executing the above command, and see the result in test_results folder. 

See the results in the `output.txt` inside `test_results/v6_mydf-tems7-natt1.0-fingtemporal-upf-PF0,1,2_A2_afl2,-1_vfl2,-1_224_r18_bs8_lr0.0001/model_best_epoch50/imbalance`
  
For testing with FakeAVCeleb:

```
python my_train.py --out_dir . --test log_tmp/v6_mydf-tems7-natt1.0-fingtemporal-upf-PF0,1,2_A2_afl2,-1_vfl2,-1_224_r18_bs8_lr0.0001/model/model_best_epoch50.pth.tar --with_att --temporal_size 7 --dataset fakeavceleb
python my_test.py --folder test_results/v6_mydf-tems7-natt1.0-fingtemporal-upf-PF0,1,2_A2_afl2,-1_vfl2,-1_224_r18_bs8_lr0.0001/model_best_epoch50/balance --dataset fakeavceleb
```
See the results in the `output.txt` inside `test_results/v6_mydf-tems7-natt1.0-fingtemporal-upf-PF0,1,2_A2_afl2,-1_vfl2,-1_224_r18_bs8_lr0.0001/model_best_epoch50/balance`

# Reference
If you use the code, please cite the paper -
```
@InProceedings{astrid2025audiovisual,
  author       = "Astrid, Marcella and Ghorbel, Enjie and Aouada, Djamila",
  title        = "Audio-visual Deepfake Detection With Local Temporal Inconsistencies",
  booktitle    = "International Conference on Acoustics, Speech, and Signal Processing (ICASSP)",
  year         = "2025",
}
```
# Acknowledgements
Thanks to the code available at https://github.com/abhinavdhall/deepfake/tree/main/ACM_MM_2020, https://github.com/TengdaHan/DPC and https://github.com/joonson/syncnet_python.  
  
