package com.evayInfo.Inglory.DL4jDiary.dl4jExamples.dataExamples;


import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Created by sunlu on 17/12/6.
 * Lecture 7 | Import a Keras Neural Net Model into Deeplearning4j
 * https://www.youtube.com/watch?v=bI1aR1Tj2DM&list=PLockSG2gRivuxy5J_wxix3KYNZVrh8FaY&index=8
 * https://gist.github.com/tomthetrainer/f6e073444286e5d97d976bd77292a064
 */

public class ImportKerasConfig {
    private static Logger log = LoggerFactory.getLogger(ImportKerasConfig.class);

    public static void main(String[] args) throws Exception {

        MultiLayerNetwork model =
                org.deeplearning4j.nn.modelimport.keras.Model.importSequentialModel
                        ("/Users/tomhanlon/tensorflow/video/iris_model_json","/Users/tomhanlon/tensorflow/video/iris_model_save");

        int numLinesToSkip = 0;
        String delimiter = ",";
        // Read the iris.txt file as a collection of records
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

        // label index
        int labelIndex = 4;
        // num of classes
        int numClasses = 3;
        // batchsize all
        int batchSize = 150;

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);

        DataSet allData = iterator.next();
        allData.shuffle();

        // Have our model
        //we have our Record Reader to read data
        // Evaluate the model

        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(allData.getFeatureMatrix());
        eval.eval(allData.getLabels(),output);
        log.info(eval.stats());








    }

}