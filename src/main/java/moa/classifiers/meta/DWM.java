package moa.classifiers.meta;


import java.util.ArrayList;
import java.util.List;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.Measurement;
import moa.options.ClassOption;

//import moa.options.FloatOption;
import com.github.javacliparser.FloatOption;
//import moa.options.IntOption;
import com.github.javacliparser.IntOption;
//import weka.core.Instance;
import com.yahoo.labs.samoa.instances.Instance;

//import weka.core.Utils;
import moa.MOAObject;
import moa.core.Utils;

/*
 * For i = 1 ... N 													<- For every training instance
 *    weightedPredictions[] = {0, ...} 								<- Zero for every class weighted prediction
 *    For j = 1 ... M 												<- For every expert...
 *    	localPrediction = experts[j].classify(instance[i])			<- Get local prediction from expert j
 *    	If(localPrediction <> instance[i].class && i % period == 0)	<- Incorrect classification and time to update
 *    		experts[j].weight = beta * experts[j].weight			<- Reduce j weight using parameter beta
 *    	weightedPredictions[localPrediction] += experts[j].weight	<- Accumulate j weight in its slot on w.predic.
 *    globalPrediction = argmax(weightedPredictions)				<- Assume local with highest weight the Global
 *    If(i % period == 0)											<- Time to update... 
 *    	normalizeWeights(experts)									<- Scale weights such that the higher will be 1
 *    	deleteExperts(threshold)									<- Delete experts with weight below 'threshold'
 *    	If(globalPrediction <> instance[i].class)					<- Incorrect global prediction...
 *    		experts.createExpert(1)									<- Add new expert with weight = 1
 *    For j = 1...M													<- For every expert...
 *    	experts[j].train(instance[i])								<- Train them with the new instance
 *    Return globalPrediction										<- Return the globalPrediction
 */

public class DWM extends AbstractClassifier {

	private static final long serialVersionUID = 1L;
	private int addedExperts = 0;
	private int deletedExperts = 0;
	
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', 
			"Classifier to train.", Classifier.class, "bayes.NaiveBayes");
	public IntOption updatePeriodOption = new IntOption("UpdatePeriod", 'u', 
			"After how many examples it is possible to create / delete new classifiers", 5000, 1, Integer.MAX_VALUE);
	public FloatOption betaOption = new FloatOption("Beta", 'b', 
			"Multiplicative constant to have experts weight decreased", 0.5f, 0.001f, 1.0f);
	public FloatOption deleteThresholdOption = new FloatOption("DeleteThresholdOption", 'd', 
			"Experts with weight below this threshold are removed", 0.01f, 0.001f, 1.0f);
	
	public List<Expert> experts;
	protected long period = 1;
	
	//for testing only
	protected long lastExpertsSize = 0;
	
	private class Expert implements MOAObject {
		public Classifier classifier;
		public double weight = 1.0;
		public long createdAt; // when it was created
		
		Expert(long p) {
			classifier = (Classifier) getPreparedClassOption(baseLearnerOption);
			classifier.resetLearning();
			createdAt = p;
		}
		
		public String toString() {
			return "(" + createdAt + ") = " + weight;
		}

		@Override
		public int measureByteSize() {
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public MOAObject copy() {
			return null;
		}

		@Override
		public void getDescription(StringBuilder sb, int indent) {
			// TODO Auto-generated method stub
		}
	}
	
	@Override
	public void resetLearningImpl() {
		experts = new ArrayList<Expert>(); //create a new list of classifiers
		experts.add(new Expert(period)); //add first classifier to the list of experts
		addedExperts = 1;	//statistic for MOA output
		deletedExperts = 0; //statistic for MOA output
	}

	/* If a data set is hold out for training then use it to update "experts" set according to DWM rules */
	@Override
	public void trainOnInstanceImpl(Instance instance) {
		DWMTrain(instance);
	}
	
	@Override
	public double[] getVotesForInstance(Instance instance) {
		double[] votes = new double[instance.numClasses()]; //create array w/ 1 position for every class
		votes[(int)DWMTest(instance)] = 1; //global prediction=index of the class with more votes
		return votes;
	}

	private double DWMTrain(Instance instance) {
		/* Create new weighted prediction */
		double[] weightedPredictions = new double[instance.numClasses()];
		
		double totalWeight = 0.0;
		/* For every expert (j...M) */
		for(Expert e : experts) {
			/* Get local prediction from expert j */
			double localPrediction = Utils.maxIndex(e.classifier.getVotesForInstance(instance));
			/* Incorrect classification and time to update */
			if(localPrediction != instance.classValue() && period % updatePeriodOption.getValue() == 0) {
				/* Reduce expert j weight using parameter beta */
				e.weight *= betaOption.getValue();
			}
			/* Accumulate j weight in its slot on w.predic. */
			weightedPredictions[(int)localPrediction] += e.weight;
			/* Train expert with this new instance. Obs: this was moved from end of pseudocode to here */
			e.classifier.trainOnInstance(instance);
			totalWeight += e.weight;
		}
		/* Assume local with highest weight the Global Prediction*/
		double globalPrediction = Utils.maxIndex(weightedPredictions);
		/* It is time to update... */
		if(period % updatePeriodOption.getValue() == 0) {
			/* Scale weights such that the higher will be 1 (the last global prediction has the highest weight) */
			//normalize(weightedPredictions[(int)globalPrediction]);
			//normalize();
			/* Scale weights based on the total weight */
			normalize(totalWeight);
			/* deleteExperts(threshold) <- Delete experts with weight below 'threshold' */
			deleteExperts();
			/* Incorrect global prediction... */
			if(globalPrediction != instance.classValue()) {
				/* Add new expert. Its weight starts as 1 */
				experts.add(new Expert(period));
				++this.addedExperts; //statistic for MOA output
			}
		}
		++period;
		return globalPrediction;
	}
	
	private double DWMTest(Instance instance) {
		/* Create new weighted prediction */
		double[] weightedPredictions = new double[instance.numClasses()];
		/* For every expert (j...M) */
		for(Expert e : experts) {
			/* Get local prediction from expert j */
			double localPrediction = Utils.maxIndex(e.classifier.getVotesForInstance(instance));
			weightedPredictions[(int)localPrediction] += e.weight;
		}
		/* Assume local with highest weight the Global Prediction*/
		double globalPrediction = Utils.maxIndex(weightedPredictions);
		return globalPrediction;
	}
	
	/* normalizacao a partir do maior peso */
	@SuppressWarnings("unused")
	private void normalize() {
		double max = experts.get(0).weight;
		for(Expert e : experts)
			if(max < e.weight)
				max = e.weight;
		for(Expert e : experts)
			e.weight /= max;
	}
	
	/* normalizacao a partir do somatorio dos pesos */
	private void normalize(double totalWeight) {
		for(Expert e : experts)
			e.weight /= totalWeight;
	}
	
	private void deleteExperts() {
		for(int i = experts.size()-1 ; i >= 0 ; --i) {
			if(experts.get(i).weight < deleteThresholdOption.getValue()) {
				++this.deletedExperts; //statistic for MOA output
				experts.remove(i);
			}
		}
	}
	
	public void showExperts() {
		System.out.print( period + ";" + experts.size() + ";");
		for(Expert e : experts) {
			System.out.print(e.toString() + ";");
		}
		System.out.println();
	}
	
	@Override
	public boolean isRandomizable() {
		return false;
	}
	
	@Override
	public void getModelDescription(StringBuilder arg0, int arg1) {
	}
	
	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		//showExperts();
		int deleted = deletedExperts, added = addedExperts;
		deletedExperts = addedExperts = 0; //reset for further statistics
		return new Measurement[]{new Measurement("#CurrentExperts", this.experts.size()),
				new Measurement("#DeletedExperts", deleted),
				new Measurement("#AddedExperts", added)};
	}
}
