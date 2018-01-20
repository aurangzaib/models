## Ideas

- When performing augmentations, two points to be considered:
	- Does this variation occur in real world?
	- Does this variation help to reduce overfit i.e reduce network wrong attributes association with fallen and standing bottles?

- Perform data augmentation to get orientation from 0 to 360 degrees with 15 degrees step-size

- Get images on different backgrounds to remove background bias:
	- Different color of conveyor belt
	- White background color

- Get dataset of combination of bottles to remove size and color bias:
	- 0.5, 1.0, 1.5 and 2.0 litres bottles
	- Dark (Coke) and transparent (water) bottles
	- With and without labels

- Augmentation with **Additive Gaussian Noise**

- Fallen PET bottles images from Google:
	- All bottles against the wall can be used as fallen bottles
	- This will provide background unbiasing for fallen nottles
	- This provide samples for Coke, Sprite, Nestle etc
	- Metal Cans can be trained along with bottles 	