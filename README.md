# dl-offtarget

Deep Learning for off-target predictions <br>

Code Description: <br>
- *data/offtarget_createdataset.py* -> create the Bunch of the crispor data used for the experiments
- *data/offtargetcreateguideseqdataset.py* -> create the Bunch for the guide-seq experiments
- *experiments/offtargetsexp.py* -> code to reproduce the experiments with 4 x 23 & 8 x 23 encodings
- *experiments/cnns.py* -> CNNs code implementation
- *experiments/ffns.py* -> FNNs code implementation
- *experiments/mltrees.py* -> machine learning algorithms (RF, LR, NB)
- *experiments/rnns.py* -> RNNs code implementation
- *experiments/utilities.py* -> utilities function to process the data or plot graphs

Saved Models: <br>
- *saved_model_4x23* -> saved deep learning models for the predictions with 4x23 encoding. RF has a fixed random seed for the reproducibility of the results.
- *saved_model_crispr_8x23* -> saved deep learning models for the predictions with 8x23 encoding on CRISPOR data set. RF has a fixed random seed for the reproducibility of the results.
- *saved_model_guideseq_8x23* -> saved deep learning models for the predictions with 8x23 encoding on GUIDE-seq data set with transfer learning. RF has a fixed random seed for the reproducibility of the results.

Images: <br>
- *images/* -> images presented in our publication

## Reference Papers

```bibtex
@article{peng2018recognition,
  title={Recognition of CRISPR/Cas9 off-target sites through ensemble learning of uneven mismatch distributions},
  author={Peng, Hui and Zheng, Yi and Zhao, Zhixun and Liu, Tao and Li, Jinyan},
  journal={Bioinformatics},
  volume={34},
  number={17},
  pages={i757--i765},
  year={2018},
  publisher={Oxford University Press}
}
```

```bibtex
@article{lin2018off,
  title={Off-target predictions in CRISPR-Cas9 gene editing using deep learning},
  author={Lin, Jiecong and Wong, Ka-Chun},
  journal={Bioinformatics},
  volume={34},
  number={17},
  pages={i656--i663},
  year={2018},
  publisher={Oxford University Press}
}
```
