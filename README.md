# Stock Prediction

## Usage


## Example


```bash
python -m utils.train new Apple --num-days-for-predict 60 --batch-size 32 --learning-rate 0.0001 --optimizer adamax
```

```bash
python -m utils.train resume model_checkpoint.ckpt
```

```bash
python -m utils.preprocess --targets Apple --processors technical_indicators fourier_components news_features
```

## Requirements

python >= 3.10
<!-- pytorch_lightning >=  -->
<!-- tensorboard >=  -->

$$
\tilde{x}
$$