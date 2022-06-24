## Early solution for [ Google AI4Code](https://www.kaggle.com/competitions/AI4Code) competition

### Overview
This solution is based on Amet Erdem's [baseline](https://www.kaggle.com/code/aerdem4/ai4code-pytorch-distilbert-baseline) . Instead of predicting the cell position with only the markdown itself, we randomly sample up to 20 code cells to act as the global context. So your input will look something like this:
```<s> Markdown content <s> Code content 1 <s> Code content 2 <s> ... <s> Code content 20 <s> ```
### Preprocessing
To extract features for training, including the markdown-only dataframes and sampling the code cells needed for each note book, simply run:

```$ python preprocess.py```

Outputs will be in the ```./data``` folder:
```
project
│   train_mark.csv
│   train_fts.json   
|   train.csv
│   val_mark.csv
│   val_fts.json
│   val.csv
```

###  Training

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 8 --accumulation_steps 4 --epochs 5 --n_workers 8 --train_mark_path ./data/train_mark_clean.csv --val_mark_path ./data/val_mark_clean.csv```


