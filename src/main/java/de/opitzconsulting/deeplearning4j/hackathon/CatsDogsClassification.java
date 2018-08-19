package de.opitzconsulting.deeplearning4j.hackathon;

import static java.lang.Math.toIntExact;

import java.io.File;
import java.nio.file.Path;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CatsDogsClassification {

    protected static final Logger LOGGER = LoggerFactory.getLogger(CatsDogsClassification.class);
    protected static long seed = 42;
    protected static Random rng = new Random(seed);

    protected static int height = 100;
    protected static int width = 100;
    protected static int channels = 3;
    protected static int batchSize = 20;
    protected static int usedTrainImages = 3000;
    protected static int numLabels = 2;

    protected static int epochs = 25;

    private static Path TRAINING_IMAGES_DIR = DataPreprocessing.ORIGINAL_IMAGE_DIR;
    private static Path VALIDATION_DIR = DataPreprocessing.VALIDATION_DIR;
    private static String modelFileName = "model.zip";

    public static void main(String args[]) throws Exception {

        //Normalize grey values of image channels between 0 and 1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        RecordReaderDataSetIterator trainDataIter = generateRecordReaderDataSetIterator(TRAINING_IMAGES_DIR);

        MultiLayerNetwork network;
        LOGGER.info("Get network model...");
        try {
        	network = ModelSerializer.restoreMultiLayerNetwork(new File(modelFileName));
        	LOGGER.info("network restored from file...");
        }
        catch (Exception e) {
            //If model cannot be deserialized from file, a fresh model is loaded.
            network = hackathonBasicNetwork();
            LOGGER.info("new network created...");
        }

        //Visit http://localhost:9000/train to watch training progress
        startUIServer(network);

        LOGGER.info("Train model....");
        while (true) {
            trainModel(network,scaler,trainDataIter, 1);
            LOGGER.info("Evaluate model....");
            evaluateModel(scaler, network);

            LOGGER.info("Saving network to file...");
            ModelSerializer.writeModel(network, new File(modelFileName), true);
        }
    }

    private static void trainModel(MultiLayerNetwork network, DataNormalization scaler, RecordReaderDataSetIterator iter, int numEpochs) {
        scaler.fit(iter);
        iter.setPreProcessor(scaler);
        network.fit(iter, numEpochs);
    }

    private static void evaluateModel(DataNormalization scaler, MultiLayerNetwork network) throws Exception {
        DataSetIterator evalIter = generateRecordReaderDataSetIterator(VALIDATION_DIR);
        scaler.fit(evalIter);
        evalIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(evalIter);
        LOGGER.info(eval.stats(true));
    }

    private static RecordReaderDataSetIterator generateRecordReaderDataSetIterator(Path pPath) throws Exception {
        //ParentPathLabelGenerator will automatically treat image folders as output neurons
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = pPath.toFile();
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = toIntExact(fileSplit.length());

        //We expect each subdirectory as a separate class with images
        int numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length;
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, usedTrainImages);
        InputSplit[] inputSplit = fileSplit.sample(pathFilter);
        InputSplit data = inputSplit[0];

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        recordReader.initialize(data, null);
        return new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);

    }

    private static void startUIServer(MultiLayerNetwork network) {
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(100));
    }


    /**
     * ToDo: This network configuration works but it's results are not the best.
     * You can improve the network configuration by
     * researching for better image classification networks and adapt the MultiLayerConfiguration.
     *
     * You can try to improve this network or directly have a look at a better architecture.
     *
     * Recommendation:
     * You may have a look at AlexNet (https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
     *
     * Steps: Adapt Activation function, Updater (https://deeplearning4j.org/updater -> Momentum instead of easy gradient descent)
     * Add more layers, Check size of convolutional filters, etc.
     *
     * If you get stuck at this point, you can also have a look at the Deeplearning4j ModelZoo (https://deeplearning4j.org/model-zoo).
     * You can also load complete Networkconfigurations as Maven dependency in the project.
     */

    public static MultiLayerNetwork hackathonBasicNetwork() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.005)
                .activation(Activation.SIGMOID)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.0001,0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(
                        //Kernel size
                        new int[]{5, 5},
                        //Stride
                        new int[]{1, 1},
                        //Padding
                        new int[]{0, 0})
                        .name("cnn1")
                        .nIn(channels)
                        .nOut(25)
                        .biasInit((double) 0).build())
                .layer(1, new SubsamplingLayer.Builder(
                        //Kernel
                        new int[]{2, 2},
                        //Stride
                        new int[]{2, 2})
                        .name("maxpool1").build())
                .layer(2, new ConvolutionLayer.Builder(
                        //Kernel
                        new int[]{5, 5},
                        //Stride
                        new int[]{5, 5},
                        //Pad
                        new int[]{1, 1})
                        .name("cnn2")
                        .nOut(500)
                        .biasInit((double) 0).build())
                .layer(3, new SubsamplingLayer.Builder(
                        //Kernel
                        new int[]{2, 2},
                        //Stride
                        new int[]{2, 2})
                        .name("maxool2").build())
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }


}

