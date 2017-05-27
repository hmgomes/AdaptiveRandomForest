/*
 *    EvaluatePrequential.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *    @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
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
package moa.tasks;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;

import moa.classifiers.Classifier;
import moa.core.Example;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.TimingUtils;
import moa.evaluation.WindowClassificationPerformanceEvaluator;
import moa.evaluation.EWMAClassificationPerformanceEvaluator;
import moa.evaluation.FadingFactorClassificationPerformanceEvaluator;
import moa.evaluation.LearningCurve;
import moa.evaluation.LearningEvaluation;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.learners.Learner;
import moa.options.ClassOption;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import moa.streams.ExampleStream;
import com.yahoo.labs.samoa.instances.Instance;
import java.util.LinkedList;
import moa.core.InstanceExample;
import moa.core.Utils;

/**
 * Task for evaluating a classifier on a delayed stream by testing then training with each example in sequence
 * given a delay k.
 *
 * @author Heitor Murilo Gomes (hmgomes at ppgia dot pucpr dot br)
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @version $Revision: 1 $
 */
public class EvaluatePrequentialDelayed extends MainTask {

    @Override
    public String getPurposeString() {
        return "Evaluates a classifier on a stream by testing then training with each example in sequence.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption learnerOption = new ClassOption("learner", 'l',
            "Learner to train.", Classifier.class, "moa.classifiers.bayes.NaiveBayes");

    public ClassOption streamOption = new ClassOption("stream", 's',
            "Stream to learn from.", ExampleStream.class,
            "generators.RandomTreeGenerator");

    public ClassOption evaluatorOption = new ClassOption("evaluator", 'e',
            "Classification performance evaluation method.",
            LearningPerformanceEvaluator.class,
            "WindowClassificationPerformanceEvaluator");

    public IntOption delayLengthOption = new IntOption("delay", 'k',
            "Number of instances before test instance is used for training",
            1000, 1, Integer.MAX_VALUE);
    
    public IntOption initialWindowSizeOption = new IntOption("initialTrainingWindow", 'p',
        "Number of instances used for training in the beginning of the stream.",
        1000, 0, Integer.MAX_VALUE);
    
    public FlagOption trainOnInitialWindowOption = new FlagOption("trainOnInitialWindow", 'm', 
            "Whether to train or not using instances in the initial window.");
    
    public FlagOption trainInBatches = new FlagOption("trainInBatches", 'b', 
        "If set training will not be interleaved with testing. ");
    
    public IntOption instanceLimitOption = new IntOption("instanceLimit", 'i',
            "Maximum number of instances to test/train on  (-1 = no limit).",
            100000000, -1, Integer.MAX_VALUE);

    public IntOption timeLimitOption = new IntOption("timeLimit", 't',
            "Maximum number of seconds to test/train for (-1 = no limit).", -1,
            -1, Integer.MAX_VALUE);
    
    public IntOption sampleFrequencyOption = new IntOption("sampleFrequency",
            'f',
            "How many instances between samples of the learning performance.",
            100000, 0, Integer.MAX_VALUE);
    
    public IntOption memCheckFrequencyOption = new IntOption(
            "memCheckFrequency", 'q',
            "How many instances between memory bound checks.", 100000, 0,
            Integer.MAX_VALUE);

    public FileOption dumpFileOption = new FileOption("dumpFile", 'd',
            "File to append intermediate csv results to.", null, "csv", true);

    public FileOption outputPredictionFileOption = new FileOption("outputPredictionFile", 'o',
            "File to append output predictions to.", null, "pred", true);

    //New for prequential method DEPRECATED
    public IntOption widthOption = new IntOption("width",
            'w', "Size of Window", 1000);

    public FloatOption alphaOption = new FloatOption("alpha",
            'a', "Fading factor or exponential smoothing factor", .01);
    //End New for prequential methods

    protected LinkedList<Example> trainInstances;
    protected LinkedList<Integer> idTracker;
    
    @Override
    public Class<?> getTaskResultType() {
        return LearningCurve.class;
    }

    @Override
    protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
        Learner learner = (Learner) getPreparedClassOption(this.learnerOption);
        ExampleStream stream = (ExampleStream) getPreparedClassOption(this.streamOption);
        LearningPerformanceEvaluator evaluator = (LearningPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
        LearningCurve learningCurve = new LearningCurve(
                "learning evaluation instances");

        this.trainInstances = new LinkedList<Example>();
        this.idTracker = new LinkedList<Integer>();
        
        //New for prequential methods
        if (evaluator instanceof WindowClassificationPerformanceEvaluator) {
            //((WindowClassificationPerformanceEvaluator) evaluator).setWindowWidth(widthOption.getValue());
            if (widthOption.getValue() != 1000) {
                System.out.println("DEPRECATED! Use EvaluatePrequential -e (WindowClassificationPerformanceEvaluator -w " + widthOption.getValue() + ")");
                 return learningCurve;
            }
        }
        if (evaluator instanceof EWMAClassificationPerformanceEvaluator) {
            //((EWMAClassificationPerformanceEvaluator) evaluator).setalpha(alphaOption.getValue());
            if (alphaOption.getValue() != .01) {
                System.out.println("DEPRECATED! Use EvaluatePrequential -e (EWMAClassificationPerformanceEvaluator -a " + alphaOption.getValue() + ")");
                return learningCurve;
            }
        }
        if (evaluator instanceof FadingFactorClassificationPerformanceEvaluator) {
            //((FadingFactorClassificationPerformanceEvaluator) evaluator).setalpha(alphaOption.getValue());
            if (alphaOption.getValue() != .01) {
                System.out.println("DEPRECATED! Use EvaluatePrequential -e (FadingFactorClassificationPerformanceEvaluator -a " + alphaOption.getValue() + ")");
                return learningCurve;
            }
        }
        //End New for prequential methods

        learner.setModelContext(stream.getHeader());
        int maxInstances = this.instanceLimitOption.getValue();
        long instancesProcessed = 0;
        int maxSeconds = this.timeLimitOption.getValue();
        int secondsElapsed = 0;
        monitor.setCurrentActivity("Evaluating learner...", -1.0);

        File dumpFile = this.dumpFileOption.getFile();
        PrintStream immediateResultStream = null;
        if (dumpFile != null) {
            try {
                if (dumpFile.exists()) {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile, true), true);
                } else {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open immediate result file: " + dumpFile, ex);
            }
        }
        //File for output predictions
        File outputPredictionFile = this.outputPredictionFileOption.getFile();
        PrintStream outputPredictionResultStream = null;
        if (outputPredictionFile != null) {
            try {
                if (outputPredictionFile.exists()) {
                    outputPredictionResultStream = new PrintStream(
                            new FileOutputStream(outputPredictionFile, true), true);
                } else {
                    outputPredictionResultStream = new PrintStream(
                            new FileOutputStream(outputPredictionFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open prediction result file: " + outputPredictionFile, ex);
            }
        }
        boolean firstDump = true;
        boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        long lastEvaluateStartTime = evaluateStartTime;
        double RAMHours = 0.0;
        
        
        while (stream.hasMoreInstances()
                && ((maxInstances < 0) || (instancesProcessed < maxInstances))
                && ((maxSeconds < 0) || (secondsElapsed < maxSeconds))) {
            
//            this.processedInstances++;
            instancesProcessed++;
            Example currentInst = stream.nextInstance();
            
//            System.out.print("Current ("+instancesProcessed+")");
            if(instancesProcessed <= this.initialWindowSizeOption.getValue()) {
                if(this.trainOnInitialWindowOption.isSet()) {
//                    System.out.print(",TrainInit");
                    learner.trainOnInstance(currentInst);
                }
                else if((this.initialWindowSizeOption.getValue() - instancesProcessed) < this.delayLengthOption.getValue()) {
                    this.trainInstances.addLast(currentInst);
                    this.idTracker.addLast((int) instancesProcessed);
                }
            }
            else {
                this.trainInstances.addLast(currentInst);
                this.idTracker.addLast((int) instancesProcessed);

//                learner.trainOnInstance(trainInst);

                if(this.delayLengthOption.getValue() < this.trainInstances.size()) {
                    if(this.trainInBatches.isSet()) {
                        // Do not train on the latest instance, otherwise
                        // it would train on k+1 instances
                        while(this.trainInstances.size() > 1) {
//                            System.out.print(",Train("+this.idTracker.removeFirst()+")");
                            Example trainInst = this.trainInstances.removeFirst();
                            learner.trainOnInstance(trainInst);
                        }
                    }
                    else {
//                        System.out.print(",Train("+this.idTracker.removeFirst()+")");
                        Example trainInst = this.trainInstances.removeFirst();
                        learner.trainOnInstance(trainInst);
                    }
                }

                // Remove class label from test instances. 
                Instance testInstance = ((Instance) currentInst.getData()).copy();
                Example testInst = new InstanceExample(testInstance);
                testInstance.setMissing(testInstance.classAttribute());
                testInstance.setClassValue(0.0);
    //          
                double[] prediction = learner.getVotesForInstance(testInst);
    //          reinstate the testInstance as it is used in evaluator.addResult
                testInstance = ((Instance) currentInst.getData()).copy();
                testInst = new InstanceExample(testInstance);

                // Output prediction
                if (outputPredictionFile != null) {
                    int trueClass = (int) ((Instance) currentInst.getData()).classValue();
                    outputPredictionResultStream.println(Utils.maxIndex(prediction) + "," + (
                     ((Instance) testInst.getData()).classIsMissing() == true ? " ? " : trueClass));
                }
                //evaluator.addClassificationAttempt(trueClass, prediction, testInst.weight());
                evaluator.addResult(testInst, prediction);
//                System.out.print(",Test("+instancesProcessed+")");
                
                if (instancesProcessed % this.sampleFrequencyOption.getValue() == 0
                        || stream.hasMoreInstances() == false) {
//                    System.out.println("Start evaluating!");
                    long evaluateTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                    double time = TimingUtils.nanoTimeToSeconds(evaluateTime - evaluateStartTime);
                    double timeIncrement = TimingUtils.nanoTimeToSeconds(evaluateTime - lastEvaluateStartTime);
                    double RAMHoursIncrement = learner.measureByteSize() / (1024.0 * 1024.0 * 1024.0); //GBs
//                    long evaluateTime = 1;
//                    double time = 1;
//                    double timeIncrement = 1;
//                    double RAMHoursIncrement = 1;
//                    System.out.println("After time and ramHours!");
                    RAMHoursIncrement *= (timeIncrement / 3600.0); //Hours
                    RAMHours += RAMHoursIncrement;
                    lastEvaluateStartTime = evaluateTime;
                    learningCurve.insertEntry(new LearningEvaluation(
                            new Measurement[]{
                                new Measurement(
                                "learning evaluation instances",
                                instancesProcessed),
                                new Measurement(
                                "evaluation time ("
                                + (preciseCPUTiming ? "cpu "
                                : "") + "seconds)",
                                time),
                                new Measurement(
                                "model cost (RAM-Hours)",
                                RAMHours)
                            },
                            evaluator, learner));
//                    System.out.println("\tMid of evaluating!");
//                    System.out.println("After inserting entry");
                    if (immediateResultStream != null) {
                        if (firstDump) {
                            immediateResultStream.println(learningCurve.headerToString());
                            firstDump = false;
                        }
                        immediateResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
                        immediateResultStream.flush();
                    }
//                    System.out.println("Finished evaluating");
                }
                if (instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                    if (monitor.taskShouldAbort()) {
                        return null;
                    }
                    long estimatedRemainingInstances = stream.estimatedRemainingInstances();
                    if (maxInstances > 0) {
                        long maxRemaining = maxInstances - instancesProcessed;
                        if ((estimatedRemainingInstances < 0)
                                || (maxRemaining < estimatedRemainingInstances)) {
                            estimatedRemainingInstances = maxRemaining;
                        }
                    }
                    monitor.setCurrentActivityFractionComplete(estimatedRemainingInstances < 0 ? -1.0
                            : (double) instancesProcessed
                            / (double) (instancesProcessed + estimatedRemainingInstances));
                    if (monitor.resultPreviewRequested()) {
                        monitor.setLatestResultPreview(learningCurve.copy());
                    }
                    secondsElapsed = (int) TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()
                            - evaluateStartTime);
                }
            }
//            System.out.println();
        }
        if (immediateResultStream != null) {
            immediateResultStream.close();
        }
        if (outputPredictionResultStream != null) {
            outputPredictionResultStream.close();
        }
        return learningCurve;
    }
}
