# Pix2Pix MNIST Dataset

This scripts create custom MNIST Dataset which can be used to trained Pix2Pix GAN model from scratch.

## Installtion

Install all the required libraries from requirements.txt file

```python
pip install -r requirements.txt
```
Will automatically installed all the required libraries 

## Usage 

```python
python3 generate_dataset.py --dataset_size 10000 --image_size 64
```
- This will generate custom MNIST dataset inside dataset folder

- It will also generate test dataset, i.e 10% of training size