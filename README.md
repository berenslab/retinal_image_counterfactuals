# **Realistic Diffusion Counterfactuals for Retinal Fundus and OCT Images**

This code repository is associated with the paper ["Generating Realistic Counterfactuals for Retinal 
Fundus and OCT Images using Diffusion Models"](https://arxiv.org/abs/2311.11629). 
It is based on the ["Diffusion Visual Counterfactual Explanations"](https://github.com/valentyn1boreiko/DVCEs)
repository. 

Here we provide the models and code to generate diffusion counterfactuals using retinal fundus images 
and OCT scans. Our fundus classifiers are trained on the task of detecting Diabetic Retinopathy detection. With these classifiers,
counterfactuals can be generated from any fundus image to the healthy class or DR class. We also provide 5-class fundus classifiers, with which counterfactuals can be generated to one of the following classes: healthy, mild, moderate, severe and proliferative. OCT classifiers are trained to classify among the classes normal, choroidal neovascularization (CNV), drusen and Diabetic Macular Edema (DME) and OCT counterfactuals can be generated to any of these 4 classes. 

## **Diffusion Counterfactuals**
<p align="center">
  <img src="counterfactuals_examples/diffusionvce_summary.png" />
</p>

## **Examples**

### Retinal Fundus Counterfactuals
<p align="center">
  <img src="counterfactuals_examples/fundus_counterfactuals_example1.png" />
</p>

<p align="center">
  <img src="counterfactuals_examples/fundus_counterfactuals_example2.png" />
</p>

### OCT counterfactuals
<p align="center">
  <img src="counterfactuals_examples/oct_counterfactuals_example1.png" />
</p>


## **Usage**

### Requirements and installations
Link to the models and requirements will be provided here soon. 
 
### Generate diffusion counterfactuals 
To generate fundus diffusion counterfactuals of the sample images provided in samples_fundus directory, run the following snippet
```
python imagenet_VCEs_eyepacs.py --config 'eyepacs_dvces_binary_cone_proj.yml' --denoise_dist_input --batch_size 5
```

To generate OCT diffusion counterfactuals of the sample OCT images provided in samples_oct directory, run the following code snippet:
```
python imagenet_VCEs_eyepacs.py --config 'oct_dvces_4class_cone_proj.yml' --denoise_dist_input --batch_size 4
```




