# Blood vessel segmentation using line operators
**PROJECT:** 

This project aims to develop an algorithm that allows to automatically segment the retinal vessels in fundus images using line operators. In this project, the effect that the variables, threshold and line length have on the automatic segmentation obtained was also tested.

**STEPS:** 

* Code implementation for line operators;
* Conclusion of the remaining image processing methods.

**FILES:** 
* [Ricci_Perfetti@2007.pdf](https://github.com/MiguelCastro3/Blood-vessel-segmentation-using-line-operators/blob/master/Ricci_Perfetti%402007.pdf) - (Digital Retinal Images for Vessel Extraction) - scientific article on which the project was based.
* [DRIVE](https://github.com/MiguelCastro3/Blood-vessel-segmentation-using-line-operators/tree/master/DRIVE) - (Digital Retinal Images for Vessel Extraction) - contains image data sets (test and training) with images for segmentation, respective mask and respective manual segmentation.
* [Imagens segmentadas](https://github.com/MiguelCastro3/Blood-vessel-segmentation-using-line-operators/tree/master/Imagens%20segmentadas) - contains the images resulting from the algorithm developed for the automatic segmentation of retinal vessels, for different thresholds and different line lengths.
* [segmentacao.py](https://github.com/MiguelCastro3/Blood-vessel-segmentation-using-line-operators/blob/master/segmentacao.py) - code with all the methods applied in the different images to select only the retinal vessels, through the line operators.

**RESULTS:** 

An example of the results obtained:
![Sem Título](https://user-images.githubusercontent.com/66881028/84937268-60fcf500-b0d3-11ea-87ee-9b9821fea0f4.png)
| Image/Metrics  | Sensitivity (%) | Specificity (%) | Accuracy (%) |
| ------------- | ------------- | ------------- | ------------- |
| 40_training  | 64.45877847208224 | 98.62850631314416	| 94.79986809950357 |  


Effects obtained with the variation of the threshold and the line length:

![threshold](https://user-images.githubusercontent.com/66881028/84935216-4b3a0080-b0d0-11ea-9913-875057e28af9.png)

![comprimento de linha](https://user-images.githubusercontent.com/66881028/84935213-4aa16a00-b0d0-11ea-8271-99ad3d4d77bf.png)
