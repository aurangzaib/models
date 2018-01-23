
### STEPS

1- Capture images

2- Resize to 640x480

3- Convert from BMP to JPG

4- Annotations using LabelImg

5- Generate CSV from XML files

6- Generate TFRecord file

7- Verify annotations using LabelImg

8- Upload on GCML:
    
    TFRecord
    Frozen weights
    Checkpoint
    Pipeline config file

9- Run training on GCML

#### EPOCHS

```epoch = n_training_samples / batch_size```

	n_training				= 79,400
	batch_size				= 24
	1 epoch 				= 3308 steps
	10 epochs 				= 33080 steps
	50 epochs 				= 165400 steps
	100 epochs				= 330800 steps
	200 epochs				= 661600 steps