import java.util.HashMap;
import org.slf4j.LoggerFactory;

import de.hpi.websci.krestel.atm.Corpus;
import org.slf4j.Logger;

public class WordEntropyThresholdCalculator {

    double[] entropy = null;
    double[] entropySorted;
    double accuracyOfCollection;
    boolean[] collectionSpecific;
    double proportionOfCollectionSpecificWords = -1;
    public WordEntropyCalculator wordEntropyCalculator = null;
    Corpus corpus = null;
    int C = 0;
    DocumentClassificationAccuracyCalculator documentClassificationAccuracyCalculator =null;
    Logger logger;
    
	public WordEntropyThresholdCalculator(Corpus corpus, int C){
		this.corpus = corpus;
		this.C = C;
		logger = LoggerFactory.getLogger(WordEntropyThresholdCalculator.class);
		wordEntropyCalculator = new WordEntropyCalculator(corpus, C);
	}

	public double precalculatedWordEntropyThreshold(int C){
		// precalculated according to the formula given in the paper
		if(C==2){
			return 0.9182958340544896;
		}
		if(C==3){
			return 0.9463946303571861;
		}
		return -1;
	}
	
	 public double getProportionOfCollectionSpecificWords(){
		 HashMap<Integer,Integer> trainwid2wid = new HashMap<Integer,Integer>();
	    	for(Integer key : corpus.wid2trainwid.keySet()){
	    		int value = corpus.wid2trainwid.get(key);
	    		trainwid2wid.put(value, key);
	    	}
		 long collectionSpecificWords = 0;
	     long collectionIndependentWords = 0;
		 for(int w=0; w< corpus.Wtrain;w++){
        	if(collectionSpecific[w]){
        		collectionSpecificWords += corpus.getTotalOccurrence(w);
        	}else{
        		collectionIndependentWords += corpus.getTotalOccurrence(w);
        	}
	     }
		 
		 proportionOfCollectionSpecificWords = (double)(collectionSpecificWords)/(double)(collectionSpecificWords+collectionIndependentWords);
		 logger.info("proportion of collection-specific words "+proportionOfCollectionSpecificWords);
		 return proportionOfCollectionSpecificWords;
	 }
	 public boolean[] adjustEntropyThreshold(double entropyThreshold){
    	if(entropy ==null){
    		entropy = wordEntropyCalculator.getWordEntropies();
    	}
        collectionSpecific=new boolean[corpus.Wtrain];
        
        logger.info("entropy threshold "+entropyThreshold);  
        int collectionSpecificWords = 0;
        int collectionIndependentWords = 0;
        for(int w=0; w< corpus.Wtrain;w++){
        	if(entropy[w]>entropyThreshold){
        		collectionSpecific[w]=false;
        		collectionIndependentWords++;
        	}else{
        		collectionSpecific[w]=true;
        		collectionSpecificWords++;
        	}
        }
        logger.info("collection-specific words in vocabulary "+collectionSpecificWords);
        logger.info("collection-independent words in vocabulary "+collectionIndependentWords);
        logger.info("proportion of collection-specific words in vocabulary "+(double)(collectionSpecificWords)/(double)(collectionSpecificWords+collectionIndependentWords));
        return collectionSpecific;
	 }
}

