import java.util.HashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.hpi.websci.krestel.atm.Corpus;
import de.hpi.websci.krestel.atm.MathUtils;
import de.hpi.websci.krestel.atm.Utils;

public class WordEntropyCalculator{
	Logger logger;
	public double[] entropy = null;
	Corpus corpus = null;
	int C = 0;
	
	public WordEntropyCalculator(Corpus corpus, int C){
		logger = LoggerFactory.getLogger(WordEntropyCalculator.class);
		this.corpus = corpus;
		this.C = C;
	}
	public double[] getWordEntropies(){
		if(entropy == null){
			entropy = calculateWordEntropies();
		}
		return entropy;
	}	

	public double[] calculateWordEntropies(){
		int W = corpus.Wtrain;
		HashMap<Integer,Integer> trainwid2wid = new HashMap<Integer,Integer>();
    	for(Integer key : corpus.wid2trainwid.keySet()){
    		int value = corpus.wid2trainwid.get(key);
    		trainwid2wid.put(value, key);
    	}
    	int[][] countWgivenC = new int[C][W]; // number of occurrences of w in c
    	int[] countWinTotal = new int[W]; // number of occurrences of word w
    	int[] countWinC = new int[C]; //number of words in c
    	double[] probC = new double[C];
    	double[] probW = new double[W];
    	double[][] probWGivenC = new double[W][C];
    	double[][] probCGivenW = new double[C][W];
    	
    	//absolute frequency of word w in collection c
    	for(int w = 0; w < W; w++){
    		for(int c = 0; c < C; c++){
    			countWgivenC[c][w] = corpus.getOccurrence(w, c);
    			countWinC[c] += countWgivenC[c][w];
    			countWinTotal[w] += countWgivenC[c][w];
    		}
    	}
    	
    	int totalNumberOfWords = MathUtils.sum(countWinC);
    	
    	// P(c)
    	for(int c = 0; c < C; c++){
    		probC[c] = (double)(countWinC[c]) / (double)(totalNumberOfWords);
    	}
    	
    	// P(w)
    	for(int w = 0; w < W; w++){
    		probW[w] = (double)(countWinTotal[w]) / (double)(totalNumberOfWords);
    	}
    	
    	// P(w|c)
    	for(int w = 0; w < W; w++){
    		for(int c = 0; c < C; c++){
    			probWGivenC[w][c] =  (double)(countWgivenC[c][w]) / (double)(countWinC[c]);
    		}
    	}

    	double[] entropy = new double[W];
    	double[] entropyWC = new double[C];
    	for(int w=0;w<W;w++){
    		for(int c = 0; c < C; c++){
    			probCGivenW[c][w] = probWGivenC[w][c]*probC[c]/probW[w];
    			// P(c|w) = P(w|c)*P(c) / P(w)
    			// p(c|w) is the probability that a randomly chosen word w is from collection c
    			entropyWC[c] = -probCGivenW[c][w]*Math.log(probCGivenW[c][w])/Math.log(2);
    		}
    		entropy[w] = MathUtils.sum(entropyWC);
    	}
    	
    	int[] argSort = Utils.argSort(entropy);
    	for(int i=0;i<W;i++){
    		int w = argSort[i];
    		entropy[w] /= Math.log(C)/Math.log(2); // map fom interval [0;ld(c)] to interval [0,1]    	
    	}
    	return entropy;
    }

	public double[][] probCGivenW(){
		int W = corpus.Wtrain;
    	int[][] countWgivenC = new int[C][W]; // number of occurrences of w in c
    	int[] countWinTotal = new int[W]; // number of occurrences of word w
    	int[] countWinC = new int[C]; //number of words in c
    	double[] probC = new double[C];
    	double[] probW = new double[W];
    	double[][] probWGivenC = new double[W][C];
    	double[][] probCGivenW = new double[C][W];
    	
    	//absolute frequency of word w in collection c
    	for(int w = 0; w < W; w++){
    		for(int c = 0; c < C; c++){
    			countWgivenC[c][w] = corpus.getOccurrence(w, c);
    			countWinC[c] += countWgivenC[c][w];
    			countWinTotal[w] += countWgivenC[c][w];
    		}
    	}
    	
    	int totalNumberOfWords = MathUtils.sum(countWinC);
    	
    	// P(c)
    	for(int c = 0; c < C; c++){
    		probC[c] = (double)(countWinC[c]) / (double)(totalNumberOfWords);
    	}
    	
    	// P(w)
    	for(int w = 0; w < W; w++){
    		probW[w] = (double)(countWinTotal[w]) / (double)(totalNumberOfWords);
    	}
    	
    	// P(w|c)
    	for(int w = 0; w < W; w++){
    		for(int c = 0; c < C; c++){
    			probWGivenC[w][c] =  (double)(countWgivenC[c][w]) / (double)(countWinC[c]);
    		}
    	}

    	for(int w=0;w<W;w++){
    		for(int c = 0; c < C; c++){
    			probCGivenW[c][w] = probWGivenC[w][c]*probC[c]/probW[w];
    			// P(c|w) = P(w|c)*P(c) / P(w)
    			// p(c|w) is the probability that a randomly chosen word w is from collection c
    		}
    	}
    	return probCGivenW; 
	}
}
