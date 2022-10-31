# Causal-Inference-on-Networked-Data


This is the implementation of our paper "[Estimating Causal Effects on Networked Observational Data via Representation Learning](https://songjiang0909.github.io/pdf/cikm_causal.pdf)", published at CIKM'22.



Data
-----------------

The original data are from [this repo](https://github.com/rguo12/network-deconfounder-wsdm20), kudos for the authors!

Due to the size limit, we put 1) data simulation code 2) the original data and 3) the simulated data in [google drive](https://drive.google.com/drive/folders/1jHjebKNSu-Kdrr-HKj73hkdpMj-1DS7-?usp=sharing).

We use `METIS` to partion a graph. If you'd like to apply it to your data, please refer to the [official package](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview). There is also a [python version](https://metis.readthedocs.io/en/latest/#installation).

How to run?
-----------------

* Step0 (data): 
	* `mkdir data` under the root folder.
	* Download the simulated data and put them under the `data` folder.

* Step1 (run):
	* `cd ./src`
	* For BC dataset: `python main.py --dataset BC`
	* For Flickr dataset: `python main.py --dataset Flickr`
	* See explanations for other arguements and parameters in `main.py`.

The prediction, evluation results and embeddings are stored under the `result` folder.



Contact
----------------------
Song Jiang <songjiang@cs.ucla.edu>


Bibtex
----------------------

```bibtex
@inproceedings{netest2022,
  title={Estimating Causal Effects on Networked Observational Data via Representation Learning},
  author={Song Jiang, Yizhou Sun},
  booktitle={Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
  year={2022}
}
```
