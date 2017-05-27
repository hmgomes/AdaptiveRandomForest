/*
 *    AdaptiveRandomForest.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Heitor Murilo Gomes (heitor_murilo_gomes@yahoo.com.br)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.Callable;

import moa.classifiers.core.driftdetection.ADWIN;
import moa.classifiers.trees.ARFHoeffdingTree;
import moa.evaluation.BasicClassificationPerformanceEvaluator;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import moa.classifiers.core.driftdetection.PageHinkleyDM;

/**
 * Adaptive Random Forest
 *
 * <p>Adaptive Random Forest (ARF). The 3 most important aspects of this 
 * ensemble classifier are: (1) inducing diversity through resampling;
 * (2) inducing diversity through randomly selecting subsets of features for 
 * node splits (see moa.classifiers.trees.ARFHoeffdingTree.java); (3) drift 
 * detectors per base tree, which cause selective resets in response to drifts.</p>
 *
 * <p>See details in:<br /> Heitor Murilo Gomes, Albert Bifet, Jesse Read, 
 * Jean Paul Barddal, Fabricio Enembreck, Bernhard Pfharinger, Geoff Holmes, 
 * Talel Abdessalem. Adaptive random forests for evolving data stream classification. 
 * In Machine Learning, DOI: 10.1007/s10994-017-5642-8, Springer, 2017.</p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Classiﬁer to train. Must be set to ARFHoeffdingTree</li>
 * <li>-s : The number of trees in the ensemble</li>
 * <li>-o : How the number of features is interpreted: 1: use value specified, 
 * 2: sqrt(#features)+1 and ignore k, 3: #features-(sqrt(#features)+1) and ignore k, 
 * 4: #features * (k / 100), 5: #features - #features * (k / 100)</li>
 * <li>-c : The size of features per split. -k corresponds to #features - k</li>
 * <li>-a : The lambda value for bagging (lambda=6 corresponds to levBag)</li>
 * <li>-j : Number of threads to be used for training</li>
 * <li>-z : Delta of Adwin change detection</li>
 * <li>-v : Delta of Adwin change detection for warnings (start training bkg learner)</li>
 * <li>-w : Whether to use prequential accuracy weighted vote</li>
 * <li>-u : Whether to use ADWIN drift detection or not, if disabled then background learner is disabled too</li>
 * <li>-q : Whether to use background learner or immediately reset learner that detected a drift.</li>
 * <li>-y : Whether to use pageHinkley instead of ADWIN for drift/warning detection.</li>
 * </ul>
 *
 * @author Heitor Murilo Gomes (heitor_murilo_gomes@yahoo.com.br)
 * @version $Revision: 1 $
 */
public class AdaptiveRandomForest extends AbstractClassifier {

    private static final long serialVersionUID = 1L;

    public ClassOption treeLearnerOption = new ClassOption("treeLearner", 'l',
            "Random Forest Tree.", ARFHoeffdingTree.class,
            "moa.classifiers.trees.ARFHoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
        "The number of trees.", 10, 1, Integer.MAX_VALUE);
    
    public IntOption kFeaturesModeOption = new IntOption("kFeaturesMode", 'o',
        "How k is interpreted. 1: use value specified, 2: sqrt(#features)+1 and ignore k, 3: #features-(sqrt(#features)+1) and ignore k, "
                + "4: #features * (k / 100), 5: #features - #features * (k / 100)", 2, 1, 5);
    
    public IntOption kFeaturesPerTreeSizeOption = new IntOption("kFeaturesPerTreeSize", 'c',
        "Number of features allowed during each split. -k = #features - k", 2, Integer.MIN_VALUE, Integer.MAX_VALUE);
    
    public FloatOption lambdaOption = new FloatOption("lambda", 'a',
        "The lambda parameter for bagging.", 6.0, 1.0, Float.MAX_VALUE);

    public IntOption numberOfJobsOption = new IntOption("numberOfJobs", 'j',
        "Total number of concurrent jobs used for processing (-1 = as much as possible, 0 = do not use multithreading)", 1, -1, Integer.MAX_VALUE);
    
    public FloatOption deltaAdwinOption = new FloatOption("deltaAdwin", 'z',
        "Delta of Adwin change detection", 0.00001, 0.0, 1.0);
    
    // It is very important that this is a greater value than deltaAdwinOption (detect changes). 
    public FloatOption deltaAdwinWarningOption = new FloatOption("deltaAdwinWarning", 'v', 
        "Delta of Adwin change detection for warnings (start training bkg learner).", 0.0001, 0.0, 1.0);
    
    public FlagOption disableWeightedVote = new FlagOption(
        "disableWeightedVote", 'w', "Whether to use prequential accuracy weighted vote.");
    
    public FlagOption disableAdwinDriftDetectionOption = new FlagOption("disableAdwinDriftDetection", 'u',
        "Whether to use ADWIN drift detection or not, if disabled then background learner is disabled too.");

    public FlagOption disableBackgroundLearnerOption = new FlagOption("disableBackgroundLearnerOption", 'q', 
        "Whether to use background learner or immediatelly reset learner that detected a drift.");
    
    public FlagOption usePageHinkleyOption = new FlagOption("usePageHinkleyOption", 'y', 
        "Whether to use pageHinkley instead of ADWIN for drift/warning detection.");
    
    protected static final int FEATURES_SQRT = 2;
    protected static final int FEATURES_SQRT_INV = 3;
    protected static final int FEATURES_PERCENT = 4;
    protected static final int FEATURES_PERCENT_INV = 5;
    
    protected static final int SINGLE_THREAD = 0;
	
    protected ARFBaseLearner[] ensemble;
    protected long instancesSeen;
    protected int kSubspaceSize;
    protected BasicClassificationPerformanceEvaluator evaluator;

    private ExecutorService executor;
    
    @Override
    public void resetLearningImpl() {
        // Init statistics
        this.instancesSeen = 0;
        this.evaluator = new BasicClassificationPerformanceEvaluator();
        int numberOfJobs;
        if(this.numberOfJobsOption.getValue() == -1) {
            numberOfJobs = Runtime.getRuntime().availableProcessors();
//            System.out.println("Available Processors = " + numberOfJobs);
        }
        else 
            numberOfJobs = this.numberOfJobsOption.getValue();
        // SINGLE_THREAD and requesting for only 1 thread are equivalent. 
        // this.executor will be null and not used...
        if(numberOfJobs != AdaptiveRandomForest.SINGLE_THREAD && numberOfJobs != 1)
            this.executor = Executors.newFixedThreadPool(numberOfJobs);
//        System.out.println("Number of threads for training = " + numberOfJobs);
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        ++this.instancesSeen;
        if(this.ensemble == null) 
            initEnsemble(instance);
        
        Collection<TrainingRunnable> trainers = new ArrayList<TrainingRunnable>();
        for (int i = 0 ; i < this.ensemble.length ; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(instance));
            InstanceExample example = new InstanceExample(instance);
            this.ensemble[i].evaluator.addResult(example, vote.getArrayRef());
            int k = MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
            if (k > 0) {
                if(this.executor != null) {
                    TrainingRunnable trainer = new TrainingRunnable(this.ensemble[i], 
                        instance, k, this.instancesSeen);
                    trainers.add(trainer);
                }
                else { // SINGLE_THREAD is in-place... 
                    this.ensemble[i].trainOnInstance(instance, k, this.instancesSeen);
                }
            }
        }
        if(this.executor != null) {
            try {
                this.executor.invokeAll(trainers);
            } catch (InterruptedException ex) {
                throw new RuntimeException("Could not call invokeAll() on training threads.");
            }
        }
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        Instance testInstance = instance.copy();
        if(this.ensemble == null) 
            initEnsemble(testInstance);
        DoubleVector combinedVote = new DoubleVector();

        for(int i = 0 ; i < this.ensemble.length ; ++i) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(testInstance));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                double acc = this.ensemble[i].evaluator.getPerformanceMeasurements()[1].getValue();
                if(! this.disableWeightedVote.isSet() && acc > 0.0) {                        
                    for(int v = 0 ; v < vote.numValues() ; ++v) {
                        vote.setValue(v, vote.getValue(v) * acc);
                    }
                }
                combinedVote.addValues(vote);
            }
        }
        return combinedVote.getArrayRef();
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder arg0, int arg1) {
        // TODO Auto-generated method stub
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        // TODO Auto-generated method stub
        return null;
    }

    protected void initEnsemble(Instance instance) {
        // Init the ensemble.
        int ensembleSize = this.ensembleSizeOption.getValue();
        this.ensemble = new ARFBaseLearner[ensembleSize];
        
        // TODO: this should be an option with default = BasicClassificationPerformanceEvaluator
//        BasicClassificationPerformanceEvaluator classificationEvaluator = (BasicClassificationPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
        BasicClassificationPerformanceEvaluator classificationEvaluator = new BasicClassificationPerformanceEvaluator();
        
        this.kSubspaceSize = this.kFeaturesPerTreeSizeOption.getValue();
  
        // The size of k depends on 2 parameters:
        // 1) kFeaturesPerTreeSizeOption
        // 2) kFeaturesModeOption
        int n = instance.numAttributes()-1; // Ignore class label ( -1 )
        double percent = this.kSubspaceSize / 100.0;

        switch(this.kFeaturesModeOption.getValue()) {
            case AdaptiveRandomForest.FEATURES_SQRT:
                this.kSubspaceSize = (int) Math.round(Math.sqrt(n)) + 1;
                break;
            case AdaptiveRandomForest.FEATURES_SQRT_INV:
                this.kSubspaceSize = n - (int) Math.round(Math.sqrt(n)) + 1;
                break;
            case AdaptiveRandomForest.FEATURES_PERCENT:
                this.kSubspaceSize = (int) Math.round(n * percent);
                break;
            case AdaptiveRandomForest.FEATURES_PERCENT_INV:
                this.kSubspaceSize = (int) Math.round(((double) n - n * percent));
                break;
        }
        // k is negative, use size(features) + -k
        if(this.kSubspaceSize < 0)
            this.kSubspaceSize = n + this.kSubspaceSize;
        // k = 0, then use at least 1
        if(this.kSubspaceSize == 0)
            this.kSubspaceSize = 1;
        // k > n, then it should use n
        if(this.kSubspaceSize > n)
            this.kSubspaceSize = n;
        
        ARFHoeffdingTree treeLearner = (ARFHoeffdingTree) getPreparedClassOption(this.treeLearnerOption);
        treeLearner.resetLearning();
        
        for(int i = 0 ; i < ensembleSize ; ++i) {
            treeLearner.subspaceSizeOption.setValue(this.kSubspaceSize);
            this.ensemble[i] = new ARFBaseLearner(
                i, 
                (ARFHoeffdingTree) treeLearner.copy(), 
                (BasicClassificationPerformanceEvaluator) classificationEvaluator.copy(), 
                this.instancesSeen, 
                ! this.disableBackgroundLearnerOption.isSet(),
                ! this.disableAdwinDriftDetectionOption.isSet(), 
                this.usePageHinkleyOption.isSet(),
                this.deltaAdwinOption.getValue(), 
                this.deltaAdwinWarningOption.getValue(), 
                false);
        }
    }
    /**
     * 
     */
    protected final class ARFBaseLearner {
        public int indexOriginal;
        public long createdOn;
        public long lastDriftOn;
        public ARFHoeffdingTree classifier;
        public boolean isBackgroundLearner;
        
        // Drift detection
        public ADWIN ADErrorDrift;
        public ADWIN ADErrorWarning;
//        public ADWINChangeDetector adwinDrift;
//        public ADWINChangeDetector adwinWarning;
        
        public boolean useBkgLearner;
        public boolean useDriftDetector;
        public double deltaAdwinDrift;
        public double deltaAdwinWarning;

        // PageHinkley
        public boolean usePageHinkley;
        protected double pageHinkleyDriftDelta = 0.01;
        protected double pageHinkleyWarningDelta = 0.005;
        public PageHinkleyDM PageHinkleyDrift;
        public PageHinkleyDM PageHinkleyWarning;
        
        // Bkg learner
        protected ARFBaseLearner bkgLearner;
        // Statistics
        public BasicClassificationPerformanceEvaluator evaluator;
        protected int numberOfDriftsDetected;
        protected int numberOfWarningsDetected;

        private void init(int indexOriginal, ARFHoeffdingTree instantiatedClassifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated, 
            long instancesSeen, boolean useBkgLearner, boolean useDriftDetector, boolean usePageHinkley, double deltaAdwinDrift, double deltaAdwinWarning, boolean isBackgroundLearner) {
            this.indexOriginal = indexOriginal;
            this.createdOn = instancesSeen;
            this.lastDriftOn = 0;
            this.classifier = instantiatedClassifier;
            this.evaluator = evaluatorInstantiated;
            this.useBkgLearner = useBkgLearner;
            this.useDriftDetector = useDriftDetector;
            this.usePageHinkley = usePageHinkley;
            this.deltaAdwinDrift = deltaAdwinDrift;
            this.deltaAdwinWarning = deltaAdwinWarning;
            this.numberOfDriftsDetected = 0;
            this.numberOfWarningsDetected = 0;
            this.isBackgroundLearner = isBackgroundLearner;

            if(this.useDriftDetector) {
                if(this.usePageHinkley) {
                    this.PageHinkleyDrift = new PageHinkleyDM();
                    this.PageHinkleyDrift.deltaOption.setValue(this.pageHinkleyDriftDelta);
                }
                else {
                    this.ADErrorDrift = new ADWIN(this.deltaAdwinDrift);
                    // Version using ADWINChangeDetector
    //                this.adwinDrift = new ADWINChangeDetector();
    //                this.adwinDrift.deltaAdwinOption.setValue(this.deltaAdwinDrift);
    //                this.adwinDrift.resetLearning();
                }
            }

            // Init Drift Detector for Warning detection. 
            if(this.useBkgLearner) {
                if(this.usePageHinkley) {
                    this.PageHinkleyWarning = new PageHinkleyDM();
                    this.PageHinkleyWarning.deltaOption.setValue(this.pageHinkleyWarningDelta);
                }
                else {
                    this.ADErrorWarning = new ADWIN(this.deltaAdwinWarning);
                    // Version using ADWINChangeDetector
    //                this.adwinWarning = new ADWINChangeDetector();
    //                this.adwinWarning.deltaAdwinOption.setValue(this.deltaAdwinWarning);
    //                this.adwinWarning.resetLearning();
                }
            }
        }

        public ARFBaseLearner(int indexOriginal, ARFHoeffdingTree instantiatedClassifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated, 
                    long instancesSeen, boolean useBkgLearner, boolean useDriftDetector, boolean usePageHinkley, double deltaAdwinDrift, double deltaAdwinWarning, boolean isBackgroundLearner) {
            init(indexOriginal, instantiatedClassifier, evaluatorInstantiated, instancesSeen, useBkgLearner, useDriftDetector, usePageHinkley, deltaAdwinDrift, deltaAdwinWarning, isBackgroundLearner);
        }

        public void reset() {
            if(this.useBkgLearner && this.bkgLearner != null) {
                this.classifier = this.bkgLearner.classifier;
                
                if(this.usePageHinkley) {
                    this.PageHinkleyWarning = this.bkgLearner.PageHinkleyWarning;
                }
                else {
                    this.ADErrorDrift = this.bkgLearner.ADErrorDrift;
                    this.ADErrorWarning = this.bkgLearner.ADErrorWarning;
                }
                this.evaluator = this.bkgLearner.evaluator;
                this.createdOn = this.bkgLearner.createdOn;
                this.bkgLearner = null;
            }
            else {
                this.classifier.resetLearning();
                this.createdOn = instancesSeen;
                if(this.usePageHinkley) {
                    this.PageHinkleyDrift = new PageHinkleyDM();
                    this.PageHinkleyDrift.deltaOption.setValue(this.pageHinkleyDriftDelta);
                }
                else {
                    this.ADErrorDrift = new ADWIN(this.deltaAdwinDrift);
                }
            }
            this.evaluator.reset();
        }

        public void trainOnInstance(Instance instance, double weight, long instancesSeen) {            
            Instance weightedInstance = (Instance) instance.copy();
            weightedInstance.setWeight(instance.weight() * weight);
            this.classifier.trainOnInstance(weightedInstance);
            
            if(this.bkgLearner != null) 
                this.bkgLearner.classifier.trainOnInstance(instance);
            
            boolean change = false, warning = false;
            // Should it use a drift detector? Also, is it a backgroundLearner? If so, then do not "incept" another one. 
            if(this.useDriftDetector && !this.isBackgroundLearner) {
                boolean correctlyClassifies = this.classifier.correctlyClassifies(instance);
                // Check for warning
                if(this.useBkgLearner) {
                    
                    if(this.usePageHinkley) {
                        // Warning PageHinkley
                        this.PageHinkleyWarning.input(correctlyClassifies ? 0 : 1);
                        if (this.PageHinkleyWarning.getChange()) {
                            System.out.println(this.indexOriginal + " PH Warning " + instancesSeen);
                            this.PageHinkleyWarning = new PageHinkleyDM();
                            this.PageHinkleyWarning.deltaOption.setValue(this.pageHinkleyWarningDelta);
                            
                            this.numberOfWarningsDetected++;
                            ARFHoeffdingTree bkgClassifier = (ARFHoeffdingTree) this.classifier.copy();
                            bkgClassifier.resetLearning();
    //                        System.out.println("ADErrorWarning " + instancesSeen + " estimation: " + this.ADErrorWarning.getEstimation());  
                            BasicClassificationPerformanceEvaluator bkgEvaluator = (BasicClassificationPerformanceEvaluator) this.evaluator.copy();
                            bkgEvaluator.reset();
                            this.bkgLearner = new ARFBaseLearner(indexOriginal, bkgClassifier, bkgEvaluator, instancesSeen, 
                                this.useBkgLearner, this.useDriftDetector, this.usePageHinkley, this.deltaAdwinDrift, this.deltaAdwinWarning, true);
                        }
                    }
                    else {
                        // Warning ADWIN
                        double ErrEstimWarning = this.ADErrorWarning.getEstimation();
                        if (this.ADErrorWarning.setInput(correctlyClassifies ? 0 : 1)) 
                            if (this.ADErrorWarning.getEstimation() > ErrEstimWarning) 
                                warning = true;
                        if(warning) {
                            this.numberOfWarningsDetected++;
                            ARFHoeffdingTree bkgClassifier = (ARFHoeffdingTree) this.classifier.copy();
                            bkgClassifier.resetLearning();
    //                        System.out.println("ADErrorWarning " + instancesSeen + " estimation: " + this.ADErrorWarning.getEstimation());  
                            BasicClassificationPerformanceEvaluator bkgEvaluator = (BasicClassificationPerformanceEvaluator) this.evaluator.copy();
                            bkgEvaluator.reset();
                            this.bkgLearner = new ARFBaseLearner(indexOriginal, bkgClassifier, bkgEvaluator, instancesSeen, 
                                this.useBkgLearner, this.useDriftDetector, this.usePageHinkley, this.deltaAdwinDrift, this.deltaAdwinWarning, true);
                            this.ADErrorWarning = new ADWIN(this.deltaAdwinDrift);
                        }
                    }
                }
                
                /*********** drift detection ***********/
                if(this.usePageHinkley) {
                    // Use PageHinkley Drift
                    this.PageHinkleyDrift.input(correctlyClassifies ? 0 : 1);
                    if (this.PageHinkleyDrift.getChange()) {
                        System.out.println(this.indexOriginal + " PH Drift " + instancesSeen);
                        this.PageHinkleyDrift = new PageHinkleyDM();
                        this.PageHinkleyDrift.deltaOption.setValue(this.pageHinkleyDriftDelta);
                        
                        this.lastDriftOn = instancesSeen;
                        this.numberOfDriftsDetected++;
                        this.reset();
                    } 
                }
                else {
                    // Use ADWIN Drift
                    double ErrEstim = this.ADErrorDrift.getEstimation();
                    if (this.ADErrorDrift.setInput(correctlyClassifies ? 0 : 1)) 
                        if (this.ADErrorDrift.getEstimation() > ErrEstim) 
                            change = true;

                    if (change) {
    //                    System.out.println("Change detected ADError at " + instancesSeen + " estimation: " + this.ADErrorDrift.getEstimation());
                        this.lastDriftOn = instancesSeen;
                        this.numberOfDriftsDetected++;
                        if (this.ADErrorDrift.getEstimation() > 0.0) { 
                            this.reset();
                        }
                    }
                }
            }
        }

        public double[] getVotesForInstance(Instance instance) {
            DoubleVector vote = new DoubleVector(this.classifier.getVotesForInstance(instance));
            return vote.getArrayRef();
        }
    }
    
    protected class TrainingRunnable implements Runnable, Callable<Integer> {
        final private ARFBaseLearner learner;
        final private Instance instance;
        final private double weight;
        final private long instancesSeen;

        public TrainingRunnable(ARFBaseLearner learner, Instance instance, 
                double weight, long instancesSeen) {
            this.learner = learner;
            this.instance = instance;
            this.weight = weight;
            this.instancesSeen = instancesSeen;
        }

        @Override
        public void run() {
            learner.trainOnInstance(this.instance, this.weight, this.instancesSeen);
        }

        @Override
        public Integer call() throws Exception {
            run();
//            throw new UnsupportedOperationException("Not supported yet."); 
            return 0;
        }
    }
}
