## Notes

- The features which remains same across the entire dataset (e.g rough shape of bottles) are what CNN ends up learning. 

- If in every standing bottle there is a cap like features then CNN will assiciate caps with standing bottles which is a good thing, but if every fallen bottle is in red color then CNN will associate red color with fallen bottle which is a bad thing.

- Following two are things to be considered:
    - Training on **single dataset** multiple times in small steps using checkpoints
    - Training on **multiple datasets** each with complete round (10 >)

- Whenever training on a new dataset, it is very important to also provide old dataset. This way CNN forms the relation between both datasets and remembers common features in both. 

- When CNN is trained on new dataset withtout mixing old dataset, it will ignore features learnt from old dataset.


## Ideas

- When performing augmentations, two points to be considered:
	- Does this variation occur in real world?
	- Does this variation help to reduce overfit i.e reduce network wrong attributes association with fallen and standing bottles?

- Perform data augmentation to get orientation from 0 to 360 degrees with 5 degrees step-size

- Get images on **different backgrounds** to remove background bias:
	- Different color of conveyor belt
	- Different background color without conveyor belt

- Get dataset of combination of bottles to **remove size and color bias**:
	- 0.5, 1.0, 1.5 and 2.0 litres bottles
	- Dark (Coke) and transparent (water) bottles
	- With and without labels
	- Damaged labels

- Augmentation with **Additive Gaussian Noise**

- Fallen PET bottles images from Google:
	- All bottles against the wall can be used as fallen bottles
	- This will provide background unbiasing for fallen bottles
	- Metal Cans can be trained along with bottles