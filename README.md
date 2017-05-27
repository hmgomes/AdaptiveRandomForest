# AdaptiveRandomForest
Repository for the AdaptiveRandomForest (also known as ARF) algorithm implemented in MOA 2016-04

The Adaptive Random Forest (ARF) algorithm is going to be available as an extension to MOA in the future. 
Until that, you may use this repository to have access to its source code or to an executable MOA-2016-04 jar. 

For more informations about MOA, check out the official website: 
http://moa.cms.waikato.ac.nz

## Citing AdaptiveRandomForest
To cite this ARF in a publication, please cite the following paper: 
> Heitor Murilo Gomes, Albert Bifet, Jesse Read, Jean Paul Barddal, Fabricio Enembreck, Bernhard Pfharinger, Geoff Holmes, Talel Abdessalem. 
> Adaptive random forests for evolving data stream classification. In Machine Learning, DOI: 10.1007/s10994-017-5642-8, Springer, 2017.

## Important source files
If you are here, then you are probably looking for the implementations used in the original AdaptiveRandomForest paper, which are:
* AdaptiveRandomForest.java: The ensemble learner AdaptiveRandomForest
* ARFHoeffdingTree.java: The base tree learner used by AdaptiveRandomForest
* EvaluatePrequentialDelayed.java: The evaluation task that includes a _k_ delay
* EvaluatePrequentialDelayedCV.java: Similar to EvaluatePrequentialDelayed.java, however it simulates Cross-validation. 

## How to execute it
To test AdaptiveRandomForest in either delayed or immediate setting execute the moa-2016.04-ARF.jar. 
You can copy and paste the following command in the interface (right click the configuration text edit and select "Enter configuration”).
Sample command: 

`EvaluatePrequentialDelayedCV -l (meta.AdaptiveRandomForest -l (ARFHoeffdingTree -e 2000000 -g 50 -c 0.01) -s 100 -o 2 -a 6.0 -z 1.0E-5 -v 1.0E-4) -s (ArffFileStream -f elecNormNew.arff) -e BasicClassificationPerformanceEvaluator -f 100000000`

Explanation: this command executes a 10 fold cross-validation delayed prequential 
evaluation on ARF with 100 classifiers (-s 100) using m = sqrt(total features) + 1 (-o 2) 
on the ELEC dataset (-f elecNormNew.arff). 
**Make sure to extract the elecNormNew.arff dataset, and setting -f to its location, before executing the command.**

## Datasets used in the original paper
The real datasets are compressed and available at the root directory. 
The synthetic datasets configurations are available at SYNTHETIC_STREAMS.txt, also on the root directory. 
