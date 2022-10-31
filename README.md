# Causal-Inference-on-Networked-Data


This is the implementation of our paper "[Estimating Causal Effects on Networked Observational Data via Representation Learning](https://songjiang0909.github.io/pdf/cikm_causal.pdf)", published at CIKM'22.



Data
-----------------

The original data are from [this repo](https://github.com/rguo12/network-deconfounder-wsdm20), kudos for the authors!

Due to the size limit, we put 1) data simulation code 2) the original data and 3) the simulated data in [google drive](https://drive.google.com/drive/folders/1jHjebKNSu-Kdrr-HKj73hkdpMj-1DS7-?usp=sharing).


How to run?
-----------------

* Step0 (data): 
	* `mkdir data` under the root folder.
	* Download the simulated data and put them under the `data` folder.

* Step1 (run):
	* `cd ./src`
	* For Aminer dataset: `python main.py --dataset aminer --epochs 700 --batch_size 3000`
	* For APS dataset: `python main.py --dataset aps --epochs 500 --batch_size 1200`
