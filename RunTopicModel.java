import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.hpi.websci.krestel.cclda.TopicModelCcLDA;

public class RunTopicModel {

	static Logger logger = LoggerFactory.getLogger(RunTopicModel.class);
    final static int defaultTopicNumber = 30;

    public static void main(String[] args) {
    	int numberOfRuns = 1;
    	for(int i=1;i<=numberOfRuns;i++){
    		logger.info("run "+i);
    		run(args);
    	}
    }
	public static void run(String[] args) {

		//configuration 0=no;1=yes
		int removeStopWords = 1;
		// use tfidf weights instead of counts in matrix
		int useTfIdf = 0;
		// 0=no; x=all with x or less df
		int removeRareWords = 0;
		// 0=no; 1=tokenization,sentenceSplitting,lemmatization,pos-tagging; 2=?
		int usePipeline = 0;
		// base directory
		final String baseDirectory = System.getProperty("user.dir") + "/output/";


        // Usage: java de.hpi.websci.krestel.atm.RunTopicModel [corpus] [model] [topicNumber]
        Corpus corpus;
        TopicModel tm;
        int topicNumber;
        
        // percentage split
        double trainingPercentage = 0.2;
        
        // cross-validation
        //int numberOfSets = 10;

        if (args.length > 0) {
            switch (args[0]) {
                case "blogposts":
                    corpus = new CorpusBlogPostsFromFile(RunTopicModel.logger, baseDirectory, trainingPercentage);
                    break;
                	
                default:
                    corpus = new CorpusToy(RunTopicModel.logger, baseDirectory, trainingPercentage);
                    break;
            }
        } else {
            corpus = new CorpusPatents(RunTopicModel.logger, baseDirectory, trainingPercentage);
            corpus.trainingPercentage = 0.9;
            logger.info("corpusPercentage = " + Double.toString(corpus.trainingPercentage));
            corpus.testPercentage = 0.1;
        }
        corpus.init(removeStopWords, removeRareWords, useTfIdf, usePipeline);

        if (args.length > 1) {
            switch (args[1]) {
                case "cclda":
                    tm = new TopicModelCcLDA(RunTopicModel.logger, corpus);
                    if (!args[3].equals("_")) {
                    	TopicModelCcLDA.IncorporateWordFrequencies = Boolean.parseBoolean(args[3]);
                    }
                    if (!args[4].equals("_")) {
                    	//TopicModelCcLDA.IncorporateCitationData = Boolean.parseBoolean(args[4]);
                    }
                    if (!args[5].equals("_")) {
                    	TopicModelCcLDA.Iterations = Integer.parseInt(args[5]);
                    	logger.info("iterations: "+TopicModelCcLDA.Iterations);
                    }
                    if (!args[6].equals("_")) {
                    	TopicModelCcLDA.Runs = Integer.parseInt(args[6]);
                    	TopicModelCcLDA.Pi = 0.25;
                    	logger.info("runs: "+TopicModelCcLDA.Runs);
                    }
                    if (!args[7].equals("_")) {
                    	TopicModelCcLDA.PerplexityAfterEachIteration = Boolean.parseBoolean(args[7]);
                    	logger.info(""+TopicModelCcLDA.PerplexityAfterEachIteration);
                    }
                    if (!args[8].equals("_")) {
                    	TopicModelCcLDA.ProportionOfCollectionSpecificWords = Double.parseDouble(args[8]);
                    	logger.info(""+TopicModelCcLDA.ProportionOfCollectionSpecificWords);
                    }
                    TopicModelCcLDA.corpusName = args[0];
                    break;
                

                default:
                    tm = new TopicModelMalletLDA(RunTopicModel.logger, corpus);
                    break;
            }
        } else {
            tm = null;
        }

         if (args.length > 2) {
             topicNumber = Integer.parseInt(args[2]);
         } else {
            topicNumber = defaultTopicNumber;
        }

        long startTime = System.nanoTime();
        tm.init(topicNumber);
        tm.train();
        long trainTime = System.nanoTime() - startTime;
        logger.info("Training took " + Math.round(trainTime / 1e9) + " seconds.");

        tm.storeModel();
	}
}
