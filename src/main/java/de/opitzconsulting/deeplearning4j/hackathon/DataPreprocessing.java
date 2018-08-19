package de.opitzconsulting.deeplearning4j.hackathon;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.Stream;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import sun.awt.image.FileImageSource;
import sun.awt.image.JPEGImageDecoder;

public class DataPreprocessing {

	protected static final Logger LOGGER = LoggerFactory.getLogger(DataPreprocessing.class);
	private static final long SEED = 333;
	private static final double TWENTY_PERCENT = 0.2;
	private static final Random RNG = new Random(SEED);
	private static final String USER_DIR = System.getProperty("user.dir");
	private static final Path ORIGINAL_IMAGE_DIR = Paths.get(USER_DIR, "src", "main", "resources", "Images");
	private static final Path VALIDATION_DIR = Paths.get(USER_DIR, "src", "main", "resources", "ValidationImages");

	public static void main(String args[]) throws Exception {
		deleteCorruptJpegData();
		splitUpTrainingAndTestSet("Cat", "Dog");
	}

	// Some files are corrupted. Have a look at Cat\10404.jpg for example.
	// If we put them into the training pipeline, Deeplearning4j will crash during
	// training.
	// Therefore we need to validate the data and delete corrupt images, before
	// starting the training pipeline
	private static void deleteCorruptJpegData() throws IOException, URISyntaxException {
		LOGGER.info("Searching for invalid JPEG files and delete them...");
		Files.walk(ORIGINAL_IMAGE_DIR)
				.map(p -> p.toFile())
				.filter(f -> f.getName().endsWith(".jpg"))
				.parallel()
				.filter(f -> !DataPreprocessing.isValidJPEG(f))
				.forEach(f -> LOGGER.info("Deleting {} with result: {}", f, f.delete()));
		LOGGER.info("\t|---> Finished.");
	}

	// Split up test set (20%) of images into separate folder.
	// Hint: You can use the functionality of the classes:
	// FileSplit, InputSplit and RandomPathFilter
	// You can find the documentation here:
	// https://deeplearning4j.org/docs/latest/datavec-overview
	private static void splitUpTrainingAndTestSet(String first, String second, String... more) {
		Stream.concat(Arrays.stream(new String[] {first, second}), Arrays.stream(more))
			.forEach(clazz -> handleClass(clazz));
	}

	private static void handleClass(String pClass) {
		Path validationClassDir = VALIDATION_DIR.resolve(pClass);
		
		if(!containsFiles(validationClassDir)) {
			RandomPathFilter pathFilter = new RandomPathFilter(RNG, "jpg");
			FileSplit fs = new FileSplit(ORIGINAL_IMAGE_DIR.resolve(pClass).toFile());
			InputSplit[] is = fs.sample(pathFilter, getPercentage(TWENTY_PERCENT));
	
			if (is.length > 0) {
				// create Folder Structure for validation directory
				validationClassDir.toFile().mkdirs();
				
				// move files
				Arrays.asList(is[0].locations())
					.stream()
					.map(uri -> Paths.get(uri))
					.forEach(p -> {
									Path target = validationClassDir.resolve(p.getFileName());
									try {
										Files.move(p, target);
										LOGGER.info("Moved {} to {}", p, target);
									} catch (IOException e) {
										LOGGER.error("{} could not be copied to {}", p, target);
									}
								  });
			}
		} else {
			LOGGER.error("{} already contains files for testing the network", validationClassDir);
		}
	}
	
	private static boolean isValidJPEG(File pFile) {
		boolean isValid = false;

		try (FileInputStream fis = new FileInputStream(pFile)) {
			JPEGImageDecoder decoder = new JPEGImageDecoder(new FileImageSource(pFile.getAbsolutePath()), fis);
			decoder.produceImage();
			isValid = true;
		} catch (FileNotFoundException e) {
			LOGGER.error("{} could not be found", pFile);
		} catch (Exception e) {
			LOGGER.error("{} could not be opened", pFile);
		}

		return isValid;
	}
	
	private static boolean containsFiles(Path pPath) {
		try {
			return Files.walk(pPath).map(p -> p.toFile()).filter(f -> !f.isDirectory()).count() > 0;
		} catch (IOException e) {
			return false;
		}
	}

	private static double[] getPercentage(double percent) {
		if (percent > 1.0 || percent <= 0) {
			throw new IllegalArgumentException("percent must be between 0 and 1");
		}

		return new double[] { percent, (1.0 - percent) };
	}
}
