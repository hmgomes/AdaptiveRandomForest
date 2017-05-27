/*
 *    OzaBagPerceptronVote.java
 *    Copyright (C) 2010 University of Waikato, Hamilton, New Zealand
 *    @author Albert Bifet
 *    @author Eibe Frank
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package moa.classifiers.meta;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.core.driftdetection.ADWIN;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.core.Utils;
import moa.options.ClassOption;
import com.github.javacliparser.IntOption;
import moa.classifiers.trees.HoeffdingTree;


public class OzaBagPerceptronVote extends AbstractClassifier {


    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "moa.classifiers.trees.HoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);

    public FloatOption weightShrinkOption = new FloatOption("weightShrink", 'w',
            "The number to multiply the weight misclassified counts.", 0.5, 0.0, Float.MAX_VALUE);

    public FloatOption poissonOption = new FloatOption("poisson", 'n',
            "The number to use to compute the weight of new instances.", 6, 0.0, Float.MAX_VALUE);
    
    public FloatOption deltaAdwinOption = new FloatOption("deltaAdwin", 'a',
            "Delta of Adwin change detection", 0.002 , 0.0, 1.0);

    public FloatOption oddsOffsetOption = new FloatOption("oddsOffset", 'o',
            "Offset for odds to avoid probabilities that are zero.", 0.001 , 0.0, Float.MAX_VALUE);

    public FlagOption adwinReplaceWorstClassifierOption = new FlagOption("adwinReplaceWorstClassifier", 'z',
            "When one Adwin detects change, replace worst classifier.");
    
    // HMG change
    public FlagOption useSigmoidIncrementsGracePeriod = new FlagOption("useSigmoidIncrementsGracePeriod", 'v',
        "Whether to use sigmoid based increments to grace period of trees or not.");

    protected Classifier[] ensemble;
    protected ADWIN[] ADError;
    protected int numberOfChangesDetected;
    protected int[][] matrixCodes;
    protected boolean initMatrixCodes = false;
    protected boolean initClassifiers = false;
    protected int numberAttributes = 1;
    protected int numInstances = 0;

    // HMG change
    protected long birthdays[];
    protected int processedInstances;
    
    @Override
    public void resetLearningImpl() {
        this.initClassifiers = true;
        this.reset = true;
        // HMG change
        this.birthdays = new long[this.ensembleSizeOption.getValue()];
    }


    @Override
    public void trainOnInstanceImpl(Instance inst) {
        // HMG change
        this.processedInstances++;
        
        int numClasses = inst.numClasses();
        //Init Ensemble
        if (this.initClassifiers == true){
            this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
            Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
            baseLearner.resetLearning();
            for (int i = 0; i < this.ensemble.length; i++) {
                this.ensemble[i] = baseLearner.copy();
            }
            this.ADError = new ADWIN[this.ensemble.length];
            for (int i = 0; i < this.ensemble.length; i++) {
                this.ADError[i]=new ADWIN();
            }

            this.initClassifiers = false;
        }

        boolean Change=false;
        Instance weightedInst = (Instance) inst.copy();

        //Train Perceptron
        double[][] votes = new double[this.ensemble.length+1][numClasses];
        for (int i = 0; i < this.ensemble.length; i++) {
            double[] v = new double[numClasses];
            for (int j = 0; j < v.length; j++) {
                v[j] = (double) this.oddsOffsetOption.getValue();
            }
            double[] vt = this.ensemble[i].getVotesForInstance(inst);
            double sum = Utils.sum(vt);
            if (!Double.isNaN(sum) && (sum > 0)) {
                for (int j = 0; j < vt.length; j++) {
                    vt[j] /= sum;
                }
            }  else {
                // Just in case the base learner returns NaN
                for (int k = 0; k < vt.length; k++) {
                    vt[k] = 0.0;
                }
            }
            sum = numClasses * (double) this.oddsOffsetOption.getValue();
            for (int j = 0; j < vt.length; j++) {
                v[j] += vt[j];
                sum += vt[j];
            }
            for (int j = 0; j < vt.length; j++) {
                votes[i][j] =  Math.log (v[j] / ( sum - v[j]));
            }
        }

        if (adwinReplaceWorstClassifierOption.isSet() == false) {
            //Test ensemble of classifiers
            for (int i = 0; i < this.ensemble.length; i++) {
                boolean correctlyClassifies=this.ensemble[i].correctlyClassifies(weightedInst);
                double ErrEstim=this.ADError[i].getEstimation();
                if (this.ADError[i].setInput(correctlyClassifies ? 0 : 1)){
                    numInstances = initialNumInstancesOption.getValue();
                    if (this.ADError[i].getEstimation()> ErrEstim) {
                        Change=true;
                        //Replace classifier if ADWIN has detected change
                        numberOfChangesDetected++;
                        // HMG change
                        this.birthdays[i] = this.processedInstances;
                        
                        this.ensemble[i].resetLearning();
                        this.ADError[i]=new ADWIN((double) this.deltaAdwinOption.getValue());
                        for (int ii = 0; ii < inst.numClasses(); ii++) {
                            weightAttribute[ii][i] = 0.0;// 0.2 * Math.random() - 0.1;
                        }
                    }
                }
            }
        } else {
            //Test ensemble of classifiers
            for (int i = 0; i < this.ensemble.length; i++) {
                boolean correctlyClassifies=this.ensemble[i].correctlyClassifies(weightedInst);
                double ErrEstim=this.ADError[i].getEstimation();
                if (this.ADError[i].setInput(correctlyClassifies ? 0 : 1))
                    if (this.ADError[i].getEstimation()> ErrEstim) Change=true;
            }
            //Replace classifier with higher error if ADWIN has detected change
            if (Change) {
                numberOfChangesDetected++;
                double max=0.0; int imax=-1;
                for (int i = 0; i < this.ensemble.length; i++) {
                    if (max<this.ADError[i].getEstimation()) {
                        max=this.ADError[i].getEstimation();
                        imax=i;
                    }
                }
                if (imax!=-1) {
                    this.ensemble[imax].resetLearning();
                    this.ADError[imax]=new ADWIN((double) this.deltaAdwinOption.getValue());
                    for (int ii = 0; ii < inst.numClasses(); ii++) {
                        weightAttribute[ii][imax] = 0.0;
                    }
                }
            }
        }

        trainOnInstanceImplPerceptron(inst.numClasses(), (int)inst.classValue(), votes);

        for (int i = 0; i < this.ensemble.length; i++) {
            // HMG change
            if(this.useSigmoidIncrementsGracePeriod.isSet()) {
                long age = this.processedInstances - this.birthdays[i];
                double s = 0.015;
                int midPoint = 500;
                int gracePeriod = 100 + (int) Math.round((1/(1+Math.exp(-(s*(age-midPoint)))))*100);
                //        System.out.println(age + "," + gracePeriod);
                ((HoeffdingTree) this.ensemble[i]).gracePeriodOption.setValue(gracePeriod);
            }
            
            int k = MiscUtils.poisson(this.poissonOption.getValue(), this.classifierRandom);
            //double error = this.ADError[i].getEstimation();
            //double k = !this.ensemble[i].correctlyClassifies(weightedInst) ? 1.0
            //  : (this.classifierRandom.nextDouble() < (error / (1.0 - error)) ? 1.0 : 0.0);///error);
            if (k > 0) {
                //Instance weightedInst = (Instance) inst.copy();
                weightedInst.setWeight(inst.weight() * k);
                this.ensemble[i].trainOnInstance(weightedInst);
            }
        }
    }

    public double[] getVotesForInstance(Instance inst) {
        if (this.initClassifiers == true){
            return new double[0];
        }
        int numClasses = inst.numClasses();

        int sizeEnsemble = this.ensemble.length;

        double[][] votes = new double[sizeEnsemble+1][numClasses];
        int[] bestClassifiers = new int[sizeEnsemble];
        for (int ii = 0; ii < sizeEnsemble; ii++) {
            bestClassifiers[ii] = ii;
        }
        for (int ii = 0; ii < sizeEnsemble; ii++) {
            int i = bestClassifiers[ii];
            double[] v = new double[numClasses];
            for (int j = 0; j < v.length; j++) {
                v[j] = (double) this.oddsOffsetOption.getValue();
            }
            double[] vt = this.ensemble[i].getVotesForInstance(inst);
            double sum = Utils.sum(vt);
            if (!Double.isNaN(sum) && (sum > 0)) {
                for (int j = 0; j < vt.length; j++) {
                    vt[j] /= sum;
                }
            } else {
                // Just in case the base learner returns NaN
                for (int k = 0; k < vt.length; k++) {
                    vt[k] = 0.0;
                }
            }
            sum = numClasses * (double) this.oddsOffsetOption.getValue();
            for (int j = 0; j < vt.length; j++) {
                v[j] += vt[j];
                sum += vt[j];
            }
            for (int j = 0; j < vt.length; j++) {
                votes[ii][j] = Math.log (v[j]/( sum - v[j]));
            }
        }
        return  getVotesForInstancePerceptron(votes, bestClassifiers, inst.numClasses()) ;
    }

    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub

    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[] { new Measurement("ensemble size",
                this.ensemble != null ? this.ensemble.length : 0),
                new Measurement("change detections", this.numberOfChangesDetected)
        };
    }

    @Override
    public Classifier[] getSubClassifiers() {
        return this.ensemble.clone();
    }

    //Perceptron
    public FloatOption learningRatioOption = new FloatOption("learningRatio", 'r', "Learning ratio", 1);
    public FloatOption penaltyFactorOption = new FloatOption("lambda", 'p', "Lambda", 0.0);
    public IntOption initialNumInstancesOption = new IntOption("initialNumInstances", 'i', "initialNumInstances", 10);

    protected double[][] weightAttribute;

    protected boolean reset;

    public void trainOnInstanceImplPerceptron(int numClasses, int actualClass, double[][] votes) {

        //Init Perceptron
        if (this.reset == true) {
            this.reset = false;
            this.weightAttribute = new double[numClasses][votes.length];
            for (int i = 0; i < numClasses; i++) {
                for (int j = 0; j < votes.length - 1; j++) {
                    weightAttribute[i][j] = 1.0 / (votes.length - 1.0);
                }
            }
            numInstances = initialNumInstancesOption.getValue();
        }

        // Weight decay
        double learningRatio = learningRatioOption.getValue() * 2.0 / (numInstances + (votes.length - 1) + 2.0);
        double lambda = penaltyFactorOption.getValue();
        numInstances++;

        double[] preds = new double[numClasses];

        for (int i = 0; i < numClasses; i++) {
            preds[i] = prediction(votes, i);
        }
        for (int i = 0; i < numClasses; i++) {
            double actual = (i == actualClass) ? 1.0 : 0.0;
            double delta = (actual - preds[i]) * preds[i] * (1 - preds[i]);
            for (int j = 0; j < this.ensemble.length; j++) {
                this.weightAttribute[i][j] += learningRatio * (delta * votes[j][i] - lambda * this.weightAttribute[i][j]);
            }
            this.weightAttribute[i][this.ensemble.length] += learningRatio * delta;
        }
    }

    public double predictionPruning(double[][] votes, int[] bestClassifiers, int classVal) {
        double sum = 0.0;
        for (int i = 0; i < votes.length - 1; i++) {
            sum += (double) weightAttribute[classVal][bestClassifiers[i]] * votes[i][classVal];
        }
        sum += weightAttribute[classVal][votes.length-1];
        return 1.0 / (1.0 + Math.exp(-sum));
    }

    public double prediction(double[][] votes, int classVal) {
        double sum = 0.0;
        for (int i = 0; i < votes.length - 1; i++) {
            sum += (double) weightAttribute[classVal][i] * votes[i][classVal];
        }
        sum += weightAttribute[classVal][votes.length-1];
        return 1.0 / (1.0 + Math.exp(-sum));
    }
    public double[] getVotesForInstancePerceptron(double[][] votesEnsemble, int[] bestClassifiers, int numClasses) {
        double[] votes = new double[numClasses];
        if (this.reset == false) {
            for (int i = 0; i < votes.length; i++) {
                votes[i] = predictionPruning(votesEnsemble, bestClassifiers, i);
            }
        }
        return votes;

    }


}
