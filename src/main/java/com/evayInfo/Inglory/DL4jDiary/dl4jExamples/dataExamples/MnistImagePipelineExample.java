package com.evayInfo.Inglory.DL4jDiary.dl4jExamples.dataExamples;


import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by sunlu on 17/11/24.
 */
public class MnistImagePipelineExample {

    private static Logger log = LoggerFactory.getLogger((MnistImagePipelineExample.class));

    public static void main(String[] args) throws Exception {
        /*
        image information
        28 * 28 grayscale
        grayscale implies single channel
        */
        int height = 28;
        int width = 28;
        int channels = 1;
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int batchSize = 1;
        int outputNum = 10;

        // Define the File Paths
        File trainDate = new File("/Users/sunlu/Documents/workspace/IDEA/DL4jDiary/data/mnist_png/training");
        File testData = new File("/Users/sunlu/Documents/workspace/IDEA/DL4jDiary/data/mnist_png/testing");

        // Define the FileSplit(PATH, ALLOWED FORMATS,random)
        FileSplit train = new FileSplit(trainDate, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        // Extract the parent path as the image label
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        // Initialize the record reader
        // add a listener, to extract the name
        recordReader.initialize(train);
        recordReader.setListeners(new LogRecordListener());

        // DataSet Iterator
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

        // Scale pixel values to 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        // In production you would loop through all the data
        // in this example the loop is just through 3
        // images for demonstration purposes
        for (int i = 1; i <= 3; i++) {
            DataSet ds = dataIter.next();
            System.out.println(ds);
            System.out.println(dataIter.getLabels());
        }
    }
}
