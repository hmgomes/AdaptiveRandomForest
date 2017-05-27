/*
 *    WEKAClassifier.java
 *    Copyright (C) 2009 University of Waikato, Hamilton, New Zealand
 *    @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 *    @author FracPete (fracpete at waikato dot ac dot nz)
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

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import com.github.javacliparser.IntOption;
import moa.options.WEKAClassOption;
import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;

/**
 * Class for using a classifier from WEKA.
 *
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */
public class WEKAClassifier
        extends AbstractClassifier {

    private static final long serialVersionUID = 1L;

    protected SamoaToWekaInstanceConverter instanceConverter;

    @Override
    public String getPurposeString() {
        return "Classifier from Weka";
    }
    
    public WEKAClassOption baseLearnerOption = new WEKAClassOption("baseLearner", 'l',
            "Classifier to train.", weka.classifiers.Classifier.class, "weka.classifiers.bayes.NaiveBayesUpdateable");

    public IntOption widthOption = new IntOption("width",
            'w', "Size of Window for training learner.", 0, 0, Integer.MAX_VALUE);

    public IntOption widthInitOption = new IntOption("widthInit",
            'i', "Size of first Window for training learner.", 1000, 0, Integer.MAX_VALUE);

    public IntOption sampleFrequencyOption = new IntOption("sampleFrequency",
            'f',
            "How many instances between samples of the learning performance.",
            0, 0, Integer.MAX_VALUE);

    protected Classifier classifier;

    protected int numberInstances;

    protected weka.core.Instances instancesBuffer;

    protected boolean isClassificationEnabled;

    protected boolean isBufferStoring;

    @Override
    public void resetLearningImpl() {

        try {
            //System.out.println(baseLearnerOption.getValue());
            String[] options = weka.core.Utils.splitOptions(baseLearnerOption.getValueAsCLIString());
            createWekaClassifier(options);
        } catch (Exception e) {
            System.err.println("Creating a new classifier: " + e.getMessage());
        }
        numberInstances = 0;
        isClassificationEnabled = false;
        this.isBufferStoring = true;
        this.instanceConverter = new SamoaToWekaInstanceConverter();
    }

    @Override
    public void trainOnInstanceImpl(Instance samoaInstance) {
        // Recupera a instancia e transforma no formato do WEKA
        weka.core.Instance inst = this.instanceConverter.wekaInstance(samoaInstance);
        try {
            // Se for a primeira instancia... 
            if (numberInstances == 0) {
                // Cria um buffer de instancias
                this.instancesBuffer = new weka.core.Instances(inst.dataset());
                // Se o classificador for do tipo Updateable... 
                if (classifier instanceof UpdateableClassifier) {
                    // Constroi o classifier passando o buffer recem criado
                    classifier.buildClassifier(instancesBuffer);
                    // Deixa enable para classificacao
                    this.isClassificationEnabled = true;
                // Se o classificador nao eh updateable, ativa a flag para 
                // armazenar instancias
                } else {
                    this.isBufferStoring = true;
                }
            }
            // Incrementa o numero de instancias apresentadas pela Stream
            numberInstances++;

            // Se o classificador for updateable... 
            if (classifier instanceof UpdateableClassifier) {
                // Se o numero de instancias vistas for maior que 0 (nao eh 
                // a primeira chamada a trainOnInstance). 
                if (numberInstances > 0) {
                    // Update o classifier usando a instancia
                    // Na verdade, como tem um numberInstances++ antes do if, 
                    // creio que nunca vai ocorrer de numberInstances <= 0 aqui.
                    ((UpdateableClassifier) classifier).updateClassifier(inst);
                }
            // Se o classificador nao eh updateable... 
            } else {
                // Se o numero de instancias eh igual a widthInitWindow
                if (numberInstances == widthInitOption.getValue()) {
                    // Cria o primeiro classificador
                    //Build first time Classifier
                    buildClassifier();
                    // Ativa a classificacao
                    isClassificationEnabled = true;
                    // Ativa flag para armazenar instancias
                    //Continue to store instances
                    if (sampleFrequencyOption.getValue() != 0) {
                        isBufferStoring = true;
                    }
                }
                // Se a width for 0... 
                if (widthOption.getValue() == 0) {
                    // Se estiver armazenando instances
                    //Used from SingleClassifierDrift
                    if (isBufferStoring == true) {
                        // Armazena a instance no buffer
                        instancesBuffer.add(inst);
                    }
                } else {
                    // numInstances == rest of division of instances seen so far
                    // (numberInstances) by the sample frequency...
                    //Used form WekaClassifier without using SingleClassifierDrift
                    int numInstances = numberInstances % sampleFrequencyOption.getValue();
                    // Sew o sample frequency for 0... 
                    if (sampleFrequencyOption.getValue() == 0) {
                        // Use numInstances = numberInstances... 
                        numInstances = numberInstances;
                    }
                    // Se o numInstances for 0... 
                    if (numInstances == 0) {
                        // Ative a flag para armazenar instancias... 
                        //Begin to store instances
                        isBufferStoring = true;
                    }
                    // Se estiver ativada a flag para ativar instancias e o 
                    // numero de instancias for menor ou igual a width... 
                    if (isBufferStoring == true && numInstances <= widthOption.getValue()) {
                        // Continue armazenando instancias... 
                        //Store instances
                        instancesBuffer.add(inst);
                    }
                    // Se o numero de instancias for igual ao width... 
                    if (numInstances == widthOption.getValue()) {
                        // Construa o classifier
                        //Build Classifier
                        buildClassifier();
                        // Ative a classificacao... 
                        isClassificationEnabled = true;
                        // reinicie o buffer.
                        this.instancesBuffer = new weka.core.Instances(inst.dataset());
                        System.out.println("Training!");
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Training: " + e.getMessage());
        }
    }

    public void buildClassifier() {
        try {
            if ((classifier instanceof UpdateableClassifier) == false) {
                Classifier auxclassifier = weka.classifiers.AbstractClassifier.makeCopy(classifier);
                auxclassifier.buildClassifier(instancesBuffer);
                classifier = auxclassifier;
                isBufferStoring = false;
            }
        } catch (Exception e) {
            System.err.println("Building WEKA Classifier: " + e.getMessage());
        }
    }

    @Override
    public double[] getVotesForInstance(Instance samoaInstance) {
        weka.core.Instance inst = this.instanceConverter.wekaInstance(samoaInstance);
        double[] votes = new double[inst.numClasses()];
        if (isClassificationEnabled == false) {
            for (int i = 0; i < inst.numClasses(); i++) {
                votes[i] = 1.0 / inst.numClasses();
            }
		} else {
			try {
				votes = this.classifier.distributionForInstance(inst);
			} catch (Exception e) {
				System.err.println(e.getMessage());
			}
		}
		return votes;
	}

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        if (classifier != null) {
            out.append(classifier.toString());
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        Measurement[] m = new Measurement[0];
        return m;
    }

    public void createWekaClassifier(String[] options) throws Exception {
        String classifierName = options[0];
        String[] newoptions = options.clone();
        newoptions[0] = "";
        this.classifier = weka.classifiers.AbstractClassifier.forName(classifierName, newoptions);
    }
}
