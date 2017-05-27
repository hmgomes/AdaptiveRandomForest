package moa.tasks;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Scanner;

import moa.core.ObjectRepository;
import moa.evaluation.LearningCurve;
import moa.options.ClassOption;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.MultiChoiceOption;
import com.github.javacliparser.FlagOption;

// TODO: Add rank matrix + average rank per metric
// TODO: Add Latex output
// TODO: Add option to choose between streams as rows (learners as columns) or streams as columns (learners as rows).  
// TODO: Allow multiple runs changing random seeds of learners/streams. 
// TODO: Add statistical test
// TODO: Allow user to select which measures to extract, currently these are hardcoded (see int[] metricsIndexes = {1,2,4,5,6,7};)
/**
 * Task for executing multiple runs of the same task using different learners
 * and datasets.
 *
 * @author Heitor Murilo Gomes (hmgomes at ppgia dot pucpr dot br)
 * @version $Revision: 7 $
 */
public class Experimenter extends MainTask {

    private static final long serialVersionUID = 1L;

    private final ClassOption taskOption = new ClassOption("evaluation", 't',
            "Evaluation method to use.", Task.class, "EvaluatePrequential");

    public FileOption configFileOption = new FileOption("configFile", 'c',
            "File to read configurations from", "/Users/heitor/git/moa/moa/config.txt", "txt", true);

//    public StringOption evaluationOption = new StringOption("metricsIndexes", 'm', 
//    		"Defines which metrics to be used from the evaluation method. For example, to obtain CPU "
//    		+ "time, Ram Hours, Acc, Kappa, Kappa T and Kappa M, then for EvaluatePrequentialCV = 1,2,5,6,7,8; "
//    		+ "EvaluatePrequential = 1,2,4,6,7,8;", "1,2,4,6,7,8");
    public MultiChoiceOption evaluationMethodOption = new MultiChoiceOption(
            "evaluationMethod", 'e', "Defines which evaluation method should be used",
            new String[]{"Classification Prequential", "Classification Prequential Delayed", "Classification Prequential Batch", "Classification PrequentialCV", "Classification Prequential DelayedCV" , "Regression Prequential",
                "MultiTarget Prequential", "MultiLabel Prequential"},
            new String[]{"EvaluatePrequential", "EvaluatePrequentialDelayed", "EvaluatePrequentialBatch", "EvaluatePrequentialCV", "EvaluatePrequentialDelayedCV", "EvaluatePrequentialRegression",
                "EvaluatePrequentialMultiTarget", "EvaluatePrequentialMultiTarget with a multilabel evaluator"}, 0);

    public FlagOption outputLearnersAsRows = new FlagOption("outputLearnersAsRows", 'r',
            "Output matrix has learners as rows (checked) or streams as rows (unchecked)");
// Files
    public FileOption dumpFileOption = new FileOption("dumpFile", 'd',
            "File to append intermediate csv results to.", "/Users/heitor/git/moa/moa/results.csv", "csv", true);

    protected ArrayList<String> classifiers = new ArrayList<String>();
    protected ArrayList<String> classifiersAlias = new ArrayList<String>();
    protected ArrayList<String> streams = new ArrayList<String>();
    protected ArrayList<String> streamsAlias = new ArrayList<String>();
    protected Result result;
    protected PrintStream immediateResultStream;
    protected String[] evaluationMethods = {"EvaluatePrequential", "EvaluatePrequentialDelayed", 
        "EvaluatePrequentialBatch", "EvaluatePrequentialCV", "EvaluatePrequentialDelayedCV",
        "EvaluatePrequentialRegression", "EvaluatePrequentialMultiTarget", "EvaluatePrequentialMultiTarget"};

    protected class Result {

        protected ArrayList<String> headers;
        protected String[] metrics;
        protected ArrayList<ArrayList<ArrayList<Double>>> results = new ArrayList<ArrayList<ArrayList<Double>>>();
        protected ArrayList<ArrayList<ArrayList<Double>>> ranks = new ArrayList<ArrayList<ArrayList<Double>>>();
        protected ArrayList<String> lineIdentifiers = new ArrayList<String>();

        public Result(ArrayList<String> h) {
            this.headers = h;
        }

        public ArrayList<String> getHeaders() {
            return this.headers;
        }

        public void addResultLine(String rowID, ArrayList<LearningCurve> learningCurves, int[] metricsIndexes,
                PrintStream streamResult) {
            this.lineIdentifiers.add(rowID);
            ArrayList<ArrayList<Double>> resultLine = new ArrayList<ArrayList<Double>>();

            if (this.metrics == null) {
                assert (learningCurves != null && learningCurves.size() > 0);
                this.metrics = new String[metricsIndexes.length];
                for (int i = 0; i < metricsIndexes.length; ++i) {
                    this.metrics[i] = learningCurves.get(0).getMeasurementName(metricsIndexes[i]);
                }
            }

            // For each metric... 
            for (int i = 0; i < this.metrics.length; ++i) {
                ArrayList<Double> resultPerMetric = new ArrayList<Double>();
                // For each classifier learning curve... 
                for (int c = 0; c < learningCurves.size(); ++c) {
                    double total = 0.0;
                    for (int l = 0; l < learningCurves.get(c).numEntries(); ++l) {
                        // entry or line of measure = l, index of metric = metricsIndexes[i]
                        total += learningCurves.get(c).getMeasurement(l, metricsIndexes[i]);
                    }
                    resultPerMetric.add(total / (double) learningCurves.get(c).numEntries());
                }
                resultLine.add(resultPerMetric);
            }
            this.results.add(resultLine);
            streamResult.print(toString(this.results.size() - 1, this.results));
        }

        protected String toString(int line, ArrayList<ArrayList<ArrayList<Double>>> resultMatrix) {
            StringBuilder sb = new StringBuilder();
            StringBuilder hd = new StringBuilder();
            if (line == 0) {
                sb.append("datasets\\learners,");
                hd.append(',');
                for (int m = 0; m < this.metrics.length; ++m) {
                    for (int h = 0; h < this.headers.size(); ++h) {
                        hd.append(headers.get(h));
                        hd.append(',');
                        sb.append(this.metrics[m]);
                        sb.append(',');
                    }
                }
                hd.append('\n');
                sb.append('\n');
            }
            sb.append(this.lineIdentifiers.get(line));
            sb.append(',');
            for (int metric = 0; metric < resultMatrix.get(line).size(); ++metric) {
                for (int point = 0; point < resultMatrix.get(line).get(metric).size(); ++point) {
                    sb.append(resultMatrix.get(line).get(metric).get(point));
                    sb.append(',');
                }
            }
            sb.append('\n');
            return hd.toString() + sb.toString();
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (int line = 0; line < this.results.size(); ++line) {
                sb.append(this.toString(line, this.results));
            }
            return sb.toString();
        }
    }

    @Override
    public String getPurposeString() {
        return "Evaluates a set of classifiers on a set of streams using a given configuration file.";
    }

    @Override
    public Class<?> getTaskResultType() {
        return LearningCurve.class;
    }

    @Override
    protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
        // Read input file to attributes task, classifiers and streams
        readConfigFile();
        createOutputFile();
        if (this.outputLearnersAsRows.isSet()) {
            this.result = new Result(this.streamsAlias);
        } else {
            this.result = new Result(this.classifiersAlias);
        }
        ArrayList<LearningCurve> learningCurves = null;
//    	String task = this.evaluationMethodOption.getChosenLabel();
        String taskStr = this.evaluationMethods[this.evaluationMethodOption.getChosenIndex()];

        int[] metricsIndexes = {1, 2, 4, 5, 6, 7};
        switch (this.evaluationMethodOption.getChosenIndex()) {
            case 0:
                taskStr += " -e BasicClassificationPerformanceEvaluator";
                metricsIndexes = new int[]{1, 2, 4, 5, 6, 7};
                break;
            case 1:
                taskStr += " -e BasicClassificationPerformanceEvaluator";
                metricsIndexes = new int[]{1, 2, 4, 5, 6, 7};
                break;
            case 2:
                taskStr += " -e BasicClassificationPerformanceEvaluator";
                metricsIndexes = new int[]{1, 2, 4, 5, 6, 7};
                break;
            case 3:
                taskStr += " -e BasicClassificationPerformanceEvaluator";
                metricsIndexes = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                break;
            case 4:
                taskStr += " -e BasicClassificationPerformanceEvaluator";
                metricsIndexes = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                break;
            case 5:
                taskStr += " -e BasicRegressionPerformanceEvaluator";
                metricsIndexes = new int[]{1, 2, 4, 5, 6, 7};
                break;
            case 6:
                taskStr += " -e BasicMultiTargetPerformanceEvaluator";
                metricsIndexes = new int[]{1, 2, 4, 5};
                break;
            case 7:
                taskStr += " -e BasicMultiLabelPerformanceEvaluator";
                metricsIndexes = new int[]{1, 2, 3, 4};
                break;
            default:
                break;
        }

        if (this.outputLearnersAsRows.isSet()) {
            for (int c = 0; c < this.classifiers.size(); ++c) {
                learningCurves = new ArrayList<LearningCurve>();
                for (int s = 0; s < this.streams.size(); ++s) {
                    this.taskOption.setValueViaCLIString(taskStr + " " + this.classifiers.get(c) + " " + this.streams.get(s));
                    MainTask task = (MainTask) taskOption.materializeObject(monitor, repository);
                    learningCurves.add((LearningCurve) task.doTask(monitor, repository));
                }
                this.result.addResultLine(this.classifiersAlias.get(c), learningCurves, metricsIndexes, this.immediateResultStream);
            }
        } else {
            for (int s = 0; s < this.streams.size(); ++s) {
                learningCurves = new ArrayList<LearningCurve>();
                for (int c = 0; c < this.classifiers.size(); ++c) {
                    this.taskOption.setValueViaCLIString(taskStr + " " + this.classifiers.get(c) + " " + this.streams.get(s));
                    MainTask task = (MainTask) taskOption.materializeObject(monitor, repository);
                    learningCurves.add((LearningCurve) task.doTask(monitor, repository));
                }
                this.result.addResultLine(this.streamsAlias.get(s), learningCurves, metricsIndexes, this.immediateResultStream);
            }
        }

        this.immediateResultStream.close();
        return learningCurves.get(0);
    }

    protected void readConfigFile() {
        File configFile = this.configFileOption.getFile();
        if (configFile != null) {
            try {
                Scanner scanner = new Scanner(configFile);
                boolean readingClassifiers = true;
                while (scanner.hasNextLine()) {
                    String line = scanner.nextLine();
                    if (line.equals("")) {
                        readingClassifiers = false;
                        continue;
                    }
                    String[] cmdAlias = line.split("#");
                    if (readingClassifiers) {
                        this.classifiers.add(cmdAlias[0]);
                        this.classifiersAlias.add(cmdAlias[1]);
                    } else {
                        this.streams.add(cmdAlias[0]);
                        this.streamsAlias.add(cmdAlias[1]);
                    }
                }
                scanner.close();
            } catch (FileNotFoundException ex) {
                throw new RuntimeException("Unable to open configuration file: " + this.configFileOption.getValue(), ex);
            }
        }
    }

    protected void createOutputFile() {
        File dumpFile = this.dumpFileOption.getFile();

        if (dumpFile != null) {
            try {
                if (dumpFile.exists()) {
                    this.immediateResultStream = new PrintStream(new FileOutputStream(dumpFile, true), true);
                } else {
                    this.immediateResultStream = new PrintStream(new FileOutputStream(dumpFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException("Unable to open immediate result file: " + dumpFile, ex);
            }
        }
    }
}
