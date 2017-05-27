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
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import moa.streams.ExampleStream;
import com.yahoo.labs.samoa.instances.Instance;
import java.util.LinkedList;
import moa.core.InstanceExample;
import moa.core.Utils;
import static moa.tasks.MainTask.INSTANCES_BETWEEN_MONITOR_UPDATES;

/**
 * Task for evaluating a classifier on a stream by testing then training with each example in sequence.
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @version $Revision: 7 $
 */
public class EvaluatePrequentialBatch extends MainTask {

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

    public IntOption batchIDIndexOption = new IntOption("batchIDIndex", 'r', 
        "The index of the ID used to separate between batches instead of using fixed window sizes. ", 
        0, 0, Integer.MAX_VALUE);
    
    public IntOption instanceIDIndexOption = new IntOption("instanceIDIndex", 'b', 
        "The index of the ID used to uniquely identify instances. ", 
        4, 0, Integer.MAX_VALUE);
    
    public IntOption monthIndexOption = new IntOption("monthIndexOption", 'm', 
        "The index of the month (exclusive project). ", 
        2, 0, Integer.MAX_VALUE);
    
    public IntOption yearIndexOption = new IntOption("yearIndexOption", 'y', 
        "The index of the year (exclusive project). ", 
        3, 0, Integer.MAX_VALUE);
    
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
            "File to append output predictions to.", null, "csv", true);

    //New for prequential method DEPRECATED
    public IntOption widthOption = new IntOption("width",
            'w', "Size of Window", 1000);

    public FloatOption alphaOption = new FloatOption("alpha",
            'a', "Fading factor or exponential smoothing factor", .01);
    //End New for prequential methods

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
        
        if(outputPredictionFile == null) {
            String outName = (this.streamOption.getValueAsCLIString()+"_"+this.learnerOption.getValueAsCLIString()).replaceAll("\\s+","");
            outName = outName.substring(outName.lastIndexOf('/')+1); 
            
            this.outputPredictionFileOption = new FileOption("outputPredictionFile", 'o',
            "File to append output predictions to.", "./output_"+outName+".csv", "csv", true);
            outputPredictionFile = this.outputPredictionFileOption.getFile();
        }
        
        if (outputPredictionFile != null) {
            try {
                if (outputPredictionFile.exists()) {
                    outputPredictionResultStream = new PrintStream(
                            new FileOutputStream(outputPredictionFile, true), true);
                } else {
                    outputPredictionResultStream = new PrintStream(
                            new FileOutputStream(outputPredictionFile), true);
                }
                outputPredictionResultStream.println("ID,SAFRA,month,year,prediction,P(PERF_FINAL=1),ground-truth,classifications correct (percent),Kappa Statistic (percent),Kappa Temporal Statistic (percent),Kappa M Statistic (percent)");
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
        
        int idxBatchID = this.batchIDIndexOption.getValue();
        int idxInstanceID = this.instanceIDIndexOption.getValue();
        int idxMonth = this.monthIndexOption.getValue();
        int idxYear = this.yearIndexOption.getValue();
        boolean firstBatch = true, firstInstance = true;
        int previousBatchID = -1;
        LinkedList<Example> trainInstances = new LinkedList<Example>();
        
        while (stream.hasMoreInstances()
                && ((maxInstances < 0) || (instancesProcessed < maxInstances))
                && ((maxSeconds < 0) || (secondsElapsed < maxSeconds))) {
            instancesProcessed++;
            Example currentInst = stream.nextInstance();
            Instance instance = ((Instance) currentInst.getData()).copy();
//            System.out.print("Current ("+instance.attribute(idxInstanceID).name()+":"+instance.value(idxInstanceID)+"): ");
//            System.out.print(instance.attribute(idxBatchID).name() + " = " + instance.value(idxBatchID));
            
            if(firstInstance) {
                previousBatchID = (int) instance.value(idxBatchID);
                firstInstance = false;
            }
            
            if(previousBatchID != (int) instance.value(idxBatchID) && firstBatch) {
                firstBatch = false;
            }
            
            if(previousBatchID != (int) instance.value(idxBatchID)) {
//                System.out.print(",Train(#"+trainInstances.size()+")");
                while(trainInstances.size() > 0) {
                    learner.trainOnInstance(trainInstances.removeFirst());
                }
            }
            if( !firstBatch ) {
               
                // Test on instance
//                System.out.print(",Test");
                // Remove class label from test instances. 
                Instance testInstance = ((Instance) currentInst.getData()).copy();
                Example testExample = new InstanceExample(testInstance);
//                testInstance.setMissing(testInstance.classAttribute());
//                testInstance.setClassValue(0.0);
    //          
                double[] prediction = learner.getVotesForInstance(testExample);
    //          reinstate the testInstance as it is used in evaluator.addResult
                testInstance = ((Instance) currentInst.getData()).copy();
                testExample = new InstanceExample(testInstance);
                
                evaluator.addResult(testExample, prediction);
                // Output prediction
                if (outputPredictionFile != null) {
                    int trueClass = (int) ((Instance) currentInst.getData()).classValue();
//                    Measurement[] ms = evaluator.getPerformanceMeasurements();
//                    for(int i = 0 ; i < ms.length ; ++i) {
//                        System.out.println(i+" : "+ms[i].getName());
//                    }
                    double probability = normalizePrediction(prediction).length > 1 ? normalizePrediction(prediction)[1] : -1.0;
                    outputPredictionResultStream.println(
                            instance.value(idxInstanceID)+","+
                            instance.value(idxBatchID)+","+
                            instance.value(idxMonth)+","+
                            instance.value(idxYear)+","+
                            Utils.maxIndex(prediction)+","+
                            probability+","+
                            trueClass+","+
                            evaluator.getPerformanceMeasurements()[1].getValue()+","+
                            evaluator.getPerformanceMeasurements()[2].getValue()+","+
                            evaluator.getPerformanceMeasurements()[3].getValue()+","+
                            evaluator.getPerformanceMeasurements()[4].getValue()
                        );
                    
//                    id, safra, probabilidade good, ground-truth

//                    outputPredictionResultStream.println(Utils.maxIndex(prediction) + "," + (
//                     ((Instance) testInst.getData()).classIsMissing() == true ? " ? " : trueClass));
                }
                
                
//                for(int i = 0 ; i < evaluator.getPerformanceMeasurements().length ; ++i) {
//                    System.out.println(i + " : " + evaluator.getPerformanceMeasurements()[i].getName());
//                }
//                
//                System.out.print(",acc: "+evaluator.getPerformanceMeasurements()[1].getValue());
//                System.out.print(",Test("+instancesProcessed+")");
            }
            trainInstances.addLast(currentInst);
//            System.out.print(",StoreTrain");
            
            previousBatchID = (int) instance.value(idxBatchID);
            
//            System.out.println();
            
 
            


            if (instancesProcessed % this.sampleFrequencyOption.getValue() == 0
                    || stream.hasMoreInstances() == false) {
                long evaluateTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                double time = TimingUtils.nanoTimeToSeconds(evaluateTime - evaluateStartTime);
                double timeIncrement = TimingUtils.nanoTimeToSeconds(evaluateTime - lastEvaluateStartTime);
                double RAMHoursIncrement = learner.measureByteSize() / (1024.0 * 1024.0 * 1024.0); //GBs
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

                if (immediateResultStream != null) {
                    if (firstDump) {
                        immediateResultStream.println(learningCurve.headerToString());
                        firstDump = false;
                    }
                    immediateResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
                    immediateResultStream.flush();
                }
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
        if (immediateResultStream != null) {
            immediateResultStream.close();
        }
        if (outputPredictionResultStream != null) {
            outputPredictionResultStream.close();
        }
        return learningCurve;
    }
    
    private static double[] normalizePrediction(double[] prediction) {
        double sum = 0.0;
        for (int i = 0; i < prediction.length; i++) {
            sum += prediction[i];
        }
        for (int i = 0; i < prediction.length; i++) {
            prediction[i] /= sum;
        }
        return prediction;
    }
}
