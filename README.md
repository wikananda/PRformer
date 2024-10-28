

# PRformer = PRE +Transformer

Welcome to the official repository of the PRformer paper: 

 [PRformer: Pyramidal Recurrent Transformer for Multivariate Time Series Forecasting](https://arxiv.org/abs/2408.10483)

## Introduction

The paper proposes Pyramid RNN embeddings (PRE) module, which consists of feature pyramids and multi-scale RNNs to learn embeddings of univariate time series. 

<img src=".README.assets/image-20240811170942589.png" alt="image-20240811170942589" style="zoom:67%;" />

Like iTransformer, PRE can also replace positional encoding and significantly improve the performance of existing Transformer-based time series predictors.

![image-20240811171301326](.README.assets/image-20240811171301326.png)

It achieves state-of-the-art performance on 8 real-world time series datasets, even significantly outperforming linear model predictors.

![image-20240811171100774](.README.assets/image-20240811171100774.png)

## Usage

1. Install Pytorch and other necessary dependencies.

   ```
   pip install -r requirements.txt
   ```

2. The datasets needed for PRformer can be obtained from the [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view) provided in iTransformer. Place the *unzipped folder* into the `dataset` directory, just like the pre-configured ETT directory: `dataset\ETT-small`.

   > The ETT dataset is already downloaded and configured, ready for use.

3. You can easily reproduce the results from the paper by running the provided script command. For instance, to reproduce the main results, execute the following commands:

   ```bash
   # obtaining results on Ettm1:
   bash ./scripts/multivariate_forecast/PRformer-ETTm1.sh
   
   # ETTm2
   bash ./scripts/multivariate_forecast/PRformer-ETTm2.sh
   
   # weather
   bash ./scripts/multivariate_forecast/PRformer-weather.sh
   
   # electricity
   bash ./scripts/multivariate_forecast/PRformer-electricity.sh
   
   #so on ...
   ```

## Complexity and efficiency

Thanks to PRE, the time and space complexity of PRformer grows linearly with sequence length.

![image-20240811172348560](.README.assets/image-20240811172348560.png)

Compared to the existing Transformer SOTA baseline PatchTST, PRformer significantly reduces running time and memory usage.

<img src=".README.assets/image-20240811172013800.png" alt="image-20240811172013800" style="zoom:67%;" />

## Future Work

- [ ] PRformer for other time series tasks.
- [ ] Integrating more Transformer variants.

## Citation

If you find this repo useful, please cite our paper.

```bib
@misc{yu2024prformerpyramidalrecurrenttransformer,
      title={PRformer: Pyramidal Recurrent Transformer for Multivariate Time Series Forecasting}, 
      author={Yongbo Yu and Weizhong Yu and Feiping Nie and Xuelong Li},
      year={2024},
      eprint={2408.10483},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.10483}, 
}
```

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- iTransformer(https://github.com/thuml/iTransformer)
- Reformer (https://github.com/lucidrains/reformer-pytorch)
- Informer (https://github.com/zhouhaoyi/Informer2020)
- FlashAttention (https://github.com/shreyansh26/FlashAttention-PyTorch)
- Autoformer (https://github.com/thuml/Autoformer)
- Stationary (https://github.com/thuml/Nonstationary_Transformers)
- PatchTST(https://github.com/yuqinie98/patchtst)

## Abstract

The self-attention mechanism in Transformer architecture, invariant to sequence order, necessitates positional embeddings to encode temporal order in time series prediction. We argue that this reliance on positional embeddings restricts the Transformer's ability to effectively represent temporal sequences, particularly when employing longer lookback windows. To address this, we introduce an innovative approach that combines Pyramid RNN embeddings(PRE) for univariate time series with the Transformer's capability to model multivariate dependencies. PRE, utilizing pyramidal one-dimensional convolutional layers, constructs multiscale convolutional features that preserve temporal order. Additionally, RNNs, layered atop these features, learn multiscale time series representations sensitive to sequence order. This integration into Transformer models with attention mechanisms results in significant performance enhancements. We present the PRformer, a model integrating PRE with a standard Transformer encoder, demonstrating state-of-the-art performance on various real-world datasets. This performance highlights the effectiveness of our approach in leveraging longer lookback windows and underscores the critical role of robust temporal representations in maximizing Transformer's potential for prediction tasks.