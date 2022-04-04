# FISH++ with LM - WikiText-3

## Installation

### Data
Get to the dataset root path then run:
```bash
wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip -q wikitext-103-v1.zip
cd wikitext-103
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt
cd ..
```
### Dependencies

```bash
conda install pytorch torchvision -c pytorch
```

### Train
```bash
bash pytorch/run_enwik8_base.sh train --work_dir PATH_TO_WORK_DIR
```

