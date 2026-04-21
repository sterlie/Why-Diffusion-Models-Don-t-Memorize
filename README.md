# Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training

This repository contains code for the theoretical analysis and numerical experiments for the paper [Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training](https://arxiv.org/abs/2505.17638) by T. Bonnaire, R. Urfin, G. Biroli and M. Mézard.

## Repository Structure

The repository is organized into two main directories:

### [`Experiments/`](./Experiments/)
Contains all numerical experiments and computational code:
- **Environment setup**: Conda environments and dependencies.
- **Training scripts**: Implementation of diffusion models on GMM and CelebA datasets.
- **Generation scripts**: Sample from trained models.
- **Data preprocessing**: CelebA dataset handling.
- **Model implementations**: U-Net and simple residual network architectures, and diffusion utilities.

### [`Theory/`](./Theory/)

Contains the numerical codes used to generate the figures in the theoretical section — namely, the **spectral density (Fig. 4)** and **the training of a Random Features Neural Network (Fig 5.)**.


## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{Bonnaire2025WhyDiffusionDontMemorize,
  title   = {Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training},
  author  = {Bonnaire, Tony and Urfin, Raphael and Biroli, Giulio and M{\'e}zard, Marc},
  journal = {arXiv preprint arXiv:2505.17638},
  year    = {2025},
  url     = {https://arxiv.org/abs/2505.17638}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the code or paper, please contact T. Bonnaire (tony.bonnaire@ens.fr) and/or R. Urfin (raphael.urfin@ens.fr).


## Run Guided Experiment 
pyenv shell scewl
python run_Unet_guided.py \
  -n 1024 \
  -b 64 \
  -s 32 \
  -W 128 \
  -LR 0.0001 \
  -O Adam \
  -m ../../Data/milk10/MILK10k_Training_Metadata.csv \
  -p ../../Data/milk10/MILK10.pth \
  -l skin_tone_class \
  --device mps \
  --generate