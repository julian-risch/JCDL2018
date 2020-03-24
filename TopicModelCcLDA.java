import static org.elasticsearch.common.xcontent.XContentFactory.jsonBuilder;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;

import org.elasticsearch.action.update.UpdateRequest;
import org.elasticsearch.client.Client;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.hpi.websci.krestel.atm.Corpus;
import de.hpi.websci.krestel.atm.MathUtils;
import de.hpi.websci.krestel.atm.TopicModel;
import de.hpi.websci.krestel.atm.Utils;

/**
 * cross-collection Latent Dirichlet Allocation (ccLDA) implementation following an approach of Michael Paul and his implementation at http://www.cs.jhu.edu/~mpaul/downloads/mftm.php
 */
public class TopicModelCcLDA extends TopicModel {
	
	Random r;
	
    int[][] docsZTestSet;
    int[][] docsXTestSet;

    int[][] nDZTestSet;
    PrintWriter writerSTDDEV;	
    
    final static String name = "ldaCrossCollection";
    Logger logger = LoggerFactory.getLogger(TopicModelCcLDA.class);
    WordEntropyCalculator wordEntropyCalculator;
    
    public static boolean IncorporateWordFrequencies;
    public static String corpusName;
	public static int Iterations;
    public static double Pi; // probability for sampling a topic from cited documents 
    public static boolean PerplexityAfterEachIteration;
    public static double ProportionOfCollectionSpecificWords;

	public static int Runs;
	
    public TopicCoherenceCalculator topicCoherenceCalculator;

    // public int[][] docs; is replaced by wordLists
    public int[] docsC; // docsC[d] is the collection of document d
    public int[][] docsZ; // docsZ[d][n] is the topic of the n-th word in document d
    public int[][] docsX; // docsZ[d][n] is 0 if the n-th word in document d is collection-independent, else 1

    // counts
    public int[][] nDZ; // nDZ[d][z] counts how many words in document d are from topic z
    public int[][] nZW; // nZW[z][w] counts how often topic z is assigned to word w (collection-independent)
    public int[][][] nZWc; // nZWc[z][w][c] counts how often topic z is assigned to word w for collection c
    public int[] nZ; // nZ[z] counts how many words are from topic z (collection-independent)
    public int[][] nZc; // nZc[z][c] counts how many words are from topic z for collection c
    public int[][][] nX;
    // nX[0][c][z] counts how many words from topic z for documents from collection c are sampled collection-independent
    // nX[1][c][z] counts how many words from topic z for documents from collection c are sampled collection-dependent

    public int D; // number of documents
    public int W; // number of word types, in other words vocabulary size
    public int C; // number of collections
    public int Z; // number of topics

    public double beta; // parameter for Dirichlet distribution, the prior for topic-word distributions
    //public double alpha; // parameter for Dirichlet distribution, the prior for document-topic distributions
    boolean includeStopWordsTopic;
    double[] alpha;
    double[] alphaProbabilities;
    final double defaultAlpha = 1.0;
    final double stopWordsAlpha = 1.5;
    
    double[][] myAlpha;
    double[][] myAlphaProbabilities;
    
    double accuracyOfCollection;
    boolean[] collectionSpecific;
    double entropyThreshold = 0.9;
    
    public double delta; // parameter for Dirichlet distribution, the prior for collection-specific topic-word distributions

    // parameters for Beta distribution
    public double gamma0; // the prior for belonging to the collection-independent topic
    public double gamma1; // //the prior for belonging to the collection-dependent topic 
    
    public final int accumulateTopicAssignmentsOverRuns = 10;

    public int burnInPeriod;
    public int maximumIterationCount;
    
    double[][][] sigmaC;
    
    // storage for accumulating phi over multiple iterations
    List<double[][]> phis;
    List<double[][][]> sigmaCs;
    
    HashMap<Integer,ArrayList<Integer>> didsOfCitedDocuments = new HashMap<Integer,ArrayList<Integer>>();
    

    public TopicModelCcLDA(Logger logger, Corpus corpus) {
        super(logger, name, corpus);
        r = new Random();
        topicCoherenceCalculator = new TopicCoherenceCalculator(r.nextInt()+"");
        
    }

    @Override
    public void init(int topicNumber) {
        super.init(topicNumber);
        burnInPeriod = Iterations;
        maximumIterationCount = Iterations+10; //+100

        try {
    		writerSTDDEV = new PrintWriter("stddev-"+IncorporateWordFrequencies+" "+corpusName+".txt", "UTF-8");
    	} catch (FileNotFoundException | UnsupportedEncodingException e) {
    		e.printStackTrace();
    	}
        // beta = 0.01 corresponds to a symmetric (around the middle) probability density function. 
        // The probability is highest near the edge of the support. same holds for delta
        beta = 0.01;
        delta = 0.01;
        // gamma=1.0 and gamma1=1.0 correspond to uniform distribution
        gamma0 = 1;
        gamma1 = 1;
        logger.info("pi: "+Pi);
        Z = topicNumber;
       
        includeStopWordsTopic = true;
        
        // alpha=1.0 corresponds to uniform distribution
        // Setup alpha.
        // We usually use a symmetric alpha vector, except if includeStopWordsTopic.
        // In that case, set alpha[0] to a higher value.
        alpha = new double[Z];
        alphaProbabilities = new double[Z];
        Arrays.fill(alpha, defaultAlpha);
        if (includeStopWordsTopic) {
          logger.info("Setting stop words alpha: " + stopWordsAlpha);
          alpha[0] = stopWordsAlpha;
        }
        double sumOfAlphas = MathUtils.sum(alpha);
        for(int i=0;i<alphaProbabilities.length;i++){
        	alphaProbabilities[i] = alpha[i] / sumOfAlphas; 
        }	
    }
  
    
    public TopicModelPair meanPhis(){
    	int phisSize = phis.size();
        double[][] phi = new double[Z][W];
        double[][][] sigmaC = new double[C][Z][W];
        
    	for(int z = 0; z<Z; z++){
    		for(int w = 0; w<W; w++){
    						
				for(double[][][] phiTmp : sigmaCs){
					for(int c = 0; c < C; c++){
						sigmaC[c][z][w] += phiTmp[c][z][w];
					}
    			}
    			for(double[][] phiTmp : phis){
    				phi[z][w] += phiTmp[z][w];
    			}
    			for(int c = 0; c < C; c++){
    				sigmaC[c][z][w] /= (double)phisSize;
    			}
        		phi[z][w] /= (double)phisSize;
    		}
    	}
    	return new TopicModelPair(phi, sigmaC);
    }
    
    public TopicModelTriple computeModelParams(ArrayList<ArrayList<Integer>> wordLists, ArrayList<Integer> doc2collection, boolean estimatePhi, ArrayList<String> doc2id, boolean cIsKnown) {
    	
        if(estimatePhi){
        	logger.info("Sampling for training set...");
        	
        }else{
        	logger.info("Sampling for test set...");
        }
    	
        for (int iter = 1; iter <= maximumIterationCount; iter++) {
        	if(estimatePhi){
        		logger.info("Iteration " + iter);
        	}
            
            doSampling(wordLists, doc2collection, estimatePhi,cIsKnown, doc2id);
            if(PerplexityAfterEachIteration && estimatePhi && (iter==1||iter==10||iter==25||iter==50||iter==125||iter==200||iter==250) ) {
            	double[][][] tmpSigmaC = new double[C][Z][W];
        		for(int c = 0; c < C; c++){
        			tmpSigmaC[c] = computeSigmaForCollection(c);
        		}
        		double[][] tmpPhi =  computePhi();
        		
        		initializeTestSet(corpus.getTestDocs(), corpus.getTestDocCollections());
        		double perplexity = perplexity(tmpPhi, tmpSigmaC, corpus);
        		double meanTopicCoherence = topicCoherenceCalculator.printTopicCoherencesShort(tmpPhi, tmpSigmaC, Z, C, corpus.wid2wNew, corpus.wid2trainwid);
        		logger.info("Perplexity " + perplexity);
        		logger.info("meanTopicCoherence " + meanTopicCoherence);
            }
            
            if(iter>=burnInPeriod && iter % accumulateTopicAssignmentsOverRuns == 0 && estimatePhi){
        		phis.add(computePhi());
        		double[][][] tmpSigmaCAcc = new double[C][Z][W];
        		for(int c = 0; c < C; c++){
        			tmpSigmaCAcc[c] = computeSigmaForCollection(c);
        		}
        		sigmaCs.add(tmpSigmaCAcc);
            }
        }
    
        TopicModelPair phiPair = meanPhis();
        double[][] phi = phiPair.getFirst();
        double[][][] sigmaC = phiPair.getSecond();        
        double[][] theta = null;
        if(estimatePhi){
        	theta = computeTheta(nDZ,wordLists, doc2collection);
        }
        else{
        	theta = computeTheta(nDZTestSet, wordLists, doc2collection);
        }
   
    	return new TopicModelTriple(phi, sigmaC, theta);
    }
    @Override
    public void train() {

    		ArrayList<ArrayList<Integer>> wordLists = corpus.getTrainDocs();
        	ArrayList<Integer> doc2collection = corpus.getTrainDocCollections();
        	ArrayList<String> doc2id = corpus.getTrainDocIds();
        	
        	initialize(wordLists, doc2collection, doc2id);

        	TopicModelTriple phi_theta = computeModelParams(wordLists, doc2collection, true, doc2id, true);
            this.phi = phi_theta.getLeft();
            this.sigmaC = phi_theta.getMiddle();
            this.theta = phi_theta.getRight();
            
            TopicModelPair phiPair = meanPhis();
    		double[][] phi = phiPair.getFirst();
            double[][][] sigmaC = phiPair.getSecond();      

            initializeTestSet(corpus.getTestDocs(), corpus.getTestDocCollections());
            
            
            perplexity(phi, sigmaC, corpus);
            documentClassificationAccuracy(phi,sigmaC, getTestSetThetaUnknownC());
            topicCoherenceCalculator.printTopicCoherences(phi, sigmaC, Z, C, corpus.wid2wNew, corpus.wid2trainwid);

    }
    
    public double documentClassificationAccuracy(double[][] phi, double[][][] sigmaC, double[][] theta){
    	DocumentClassificationAccuracyCalculator documentClassificationAccuracyCalculator = new DocumentClassificationAccuracyCalculator();
    	return documentClassificationAccuracyCalculator.classifyDocuments(corpus.getTestDocCollections(), corpus.getTestDocs(), phi, sigmaC, theta, C, Z, corpus, IncorporateWordFrequencies, collectionSpecific, nX);
    }
	
	@Override
    public double[][] getTestSetTheta() {
    	ArrayList<ArrayList<Integer>> wordLists = corpus.getTestDocs();
    	ArrayList<Integer> doc2collection = corpus.getTestDocCollections();
    	ArrayList<String> doc2id = corpus.getTestDocIds();
        
    	TopicModelTriple phi_theta = computeModelParams(wordLists, doc2collection, false, doc2id, true);
    	
        return phi_theta.getRight();
    }
	
    public double[][] getTestSetThetaUnknownC() {
    	ArrayList<ArrayList<Integer>> wordLists = corpus.getTestDocs();
    	ArrayList<Integer> doc2collection = corpus.getTestDocCollections();
    	ArrayList<String> doc2id = corpus.getTestDocIds();
        
    	TopicModelTriple phi_theta = computeModelParams(wordLists, doc2collection, false, doc2id, false);
    	
        return phi_theta.getRight();
    }
    
    public void initialize(ArrayList<ArrayList<Integer>> wordLists, ArrayList<Integer> doc2collection, ArrayList<String> doc2id) {
        logger.info("Initializing...");
        
        C = corpus.cid2c.size();
        D = wordLists.size();
        Z = topicNumber;
        W = corpus.Wtrain;
        sigmaCs = new ArrayList<double[][][]>();
        phis = new ArrayList<double[][]>();
        
        corpus.countOccurrences();
        wordEntropyCalculator = new WordEntropyCalculator(corpus, C);
        
        docsC = new int[D];
        int[] numberOfDocsInC = new int[C];
        int[] numberOfWordsInC = new int[C];
        Arrays.fill(numberOfDocsInC, 0);
        Arrays.fill(numberOfWordsInC, 0);
        
        for (int d = 0; d < D; d++) {
        	docsC[d] = doc2collection.get(d);
        	numberOfDocsInC[docsC[d]]++;
        	numberOfWordsInC[docsC[d]] += wordLists.get(d).size();
        }
        logger.info("C: "+C+" D:"+D+" Z:"+Z+" W:"+W);
        for (int c = 0; c < C; c++) { 
        	logger.info("#documents in collection "+c+" "+numberOfDocsInC[c]);
        	logger.info("#words in collection "+c+" "+numberOfWordsInC[c]);
        }       
		
        WordEntropyThresholdCalculator wordEntropyThresholdCalculator = new WordEntropyThresholdCalculator(corpus, C);        
        entropyThreshold = wordEntropyThresholdCalculator.precalculatedWordEntropyThreshold(C);
    	collectionSpecific = wordEntropyThresholdCalculator.adjustEntropyThreshold(entropyThreshold);
    	ProportionOfCollectionSpecificWords = wordEntropyThresholdCalculator.getProportionOfCollectionSpecificWords();
    	
        myAlpha = new double[C][];
        for(int c = 0; c < C; c++) {
        	 myAlpha[c] = new double[Z];
        }
        for(int c = 0; c < C; c++){
        	for(int z = 0; z < Z; z++) {
        		myAlpha[c][z] = 1.0/Z;
        	}
        	myAlpha[c][0] = stopWordsAlpha;
        }

        myAlphaProbabilities = new double[C][];
        for(int c = 0; c < C; c++) { // for each collection
        	logger.info(corpus.cid2c.get(c)+" has collection index "+c);
        	myAlphaProbabilities[c] = new double[Z];
        	double sumOfMyAlphasC = MathUtils.sum(myAlpha[c]);
        	for(int i=0;i<Z;i++){
        		myAlphaProbabilities[c][i] = myAlpha[c][i] / sumOfMyAlphasC; 
        	}
        }
        for(int c=0; c<C; c++){
        	logger.info("c"+c+" alpha probabilities"+Arrays.toString(myAlphaProbabilities[c]));
        }

        docsZ = new int[D][];
        docsX = new int[D][];

        nDZ = new int[D][Z];
        nZW = new int[Z][W];
        nZWc = new int[Z][W][C];
        nZ = new int[Z];
        nZc = new int[Z][C];
        nX = new int[2][C][Z];

        for (int d = 0; d < D; d++) { // for each document
            docsZ[d] = new int[wordLists.get(d).size()];
            docsX[d] = new int[wordLists.get(d).size()];
            
            logger.debug("document "+d+" consists of "+wordLists.get(d).size()+" words");
            for (int n = 0; n < wordLists.get(d).size(); n++) { // for each word in document d
                int w = corpus.getWidTrain(wordLists.get(d).get(n)); // n-th word in document d
                int c = docsC[d];
                
                int z = Utils.sampleIndexFromProbabilityArray(myAlphaProbabilities[c]);
                docsZ[d][n] = z; // assign random topic to n-th word in document d
                
                boolean x;
                if(IncorporateWordFrequencies){
                	x = collectionSpecific[w];
                }
                else{
                	 x = r.nextBoolean();
                }
                if (!x) {
                docsX[d][n] = 0; // n-th word in document d is considered collection-dependent(1) or -independent(0) 
                // update counts
                nX[0][c][z] += 1;
                }
                else{
                	docsX[d][n] = 1; // n-th word in document d is considered collection-dependent(1) or -independent(0) 
                    // update counts
                    nX[1][c][z] += 1;
                }
                nDZ[d][z] += 1;
                if (!x) {
                    nZW[z][w] += 1;
                    nZ[z] += 1;
                } else {
                    nZWc[z][w][c] += 1;
                    nZc[z][c] += 1;
                }
            }
        }
    }

    public void initializeTestSet(ArrayList<ArrayList<Integer>> wordLists, ArrayList<Integer> doc2collection) {
        
        int D = wordLists.size();
        docsZTestSet = new int[D][];

        nDZTestSet = new int[D][Z];

        for (int d = 0; d < D; d++) { // for each document
            docsZTestSet[d] = new int[wordLists.get(d).size()];
            
            for (int n = 0; n < wordLists.get(d).size(); n++) { // for each word in document d
                int c = doc2collection.get(d); // this document's collection               
                int z = Utils.sampleIndexFromProbabilityArray(myAlphaProbabilities[c]);
                docsZTestSet[d][n] = z; // assign random topic to n-th word in document d
                nDZTestSet[d][z] += 1;
            }
        }
    }
    public double perplexity(double[][] phi, double[][][] sigmaC, Corpus corpus){
    	PerplexityCalculator perplexityCalculator= new PerplexityCalculator();
    	return perplexityCalculator.perplexity(phi, sigmaC, corpus, getTestSetTheta(), ProportionOfCollectionSpecificWords, C, Z, nX,IncorporateWordFrequencies);
    }
    public void updateValue(Client client, String index, String type, String id, String field, String value){
    	UpdateRequest updateRequest = new UpdateRequest();
    	updateRequest.index(index);
    	updateRequest.type(type);
    	updateRequest.id(id);
    	try {
			updateRequest.doc(jsonBuilder()
			        .startObject()
			            .field(field, value)
			        .endObject());
		} catch (IOException e) {
			logger.info("failed updating document "+id+" in index "+index);
			e.printStackTrace();
		}
    	try {
			client.update(updateRequest).get();
		} catch (InterruptedException | ExecutionException e) {
			logger.info("failed updating document "+id+" in index "+index);
			e.printStackTrace();
		}
    }
    
    public void doSampling(ArrayList<ArrayList<Integer>> wordLists, ArrayList<Integer> doc2collection, boolean estimatePhi, boolean cIsKnown, ArrayList<String> doc2id) {
    	
        for (int d = 0; d < wordLists.size(); d++) { // for each document
        	
        	List<Integer> dids = null;
        	
            for (int n = 0; n < wordLists.get(d).size(); n++) { // for each word in document d
            	if(corpus.getWidTrain(wordLists.get(d).get(n))<0){
            		continue; // unknown word
            	}
                if(estimatePhi){
                	sample(d, n, dids, wordLists, doc2collection);
                }else{
                	if(cIsKnown){
                		sampleTestSet(d, n, dids, wordLists, doc2collection);
                	}
                	else
                	{
                		sampleTestSetUnknownC(d, n, dids, wordLists);
                	}
                }
            }
        }
    }
    
    
    public void sample(int d, int n, List<Integer> dids, ArrayList<ArrayList<Integer>> wordLists, ArrayList<Integer> doc2collection) {
    	
    	int w = corpus.getWidTrain(wordLists.get(d).get(n));
        int c = doc2collection.get(d);
        int topic = docsZ[d][n];
        int route = docsX[d][n];

        // decrement counts for document d's n-th word
        nDZ[d][topic] -= 1;
        nX[route][c][topic] -= 1;

        if (route == 0) { // collection-independent
            nZW[topic][w] -= 1;
            nZ[topic] -= 1;
        } else { // collection-dependent
            nZWc[topic][w][c] -= 1;
            nZc[topic][c] -= 1;
        }

        double betaNorm = W * beta;
        double deltaNorm = W * delta;

        // sample new value for route
        double pTotal = 0.0;
        double[] p = new double[2];

        
        double  u;
        if(IncorporateWordFrequencies){
        	if(collectionSpecific[w]){
        		route = 1;
        	}
        	else{
        		route = 0;
        	} 
        }
        else{
	        
	        // the original ccLDA approach does not consider how often word w occurs in the other collection 
	        
	        // nX[0][c][z] counts how many words from topic z for documents from collection c are sampled collection-independent
	        // nZW[z][w] counts how often topic z is assigned to word w (collection-independent)
	        // P(x=0|z,c) * P(z|w)
	        // ----- P(z) -----
	        // nZ[z] counts how many words are from topic z (collection-independent)
	        // collection-independent
	        p[0] = (nX[0][c][topic] + gamma0) * (nZW[topic][w] + beta)
	                / (nZ[topic] + betaNorm);
	
	        // depending on the (probable) topic z of word w and the collection of word w's document, 
	        // there is a lower or higher probability P(x=0)
	        
	        // use P(w|c=0) and P(w|c=1) to sample x. 
	        // if difference of P(w|c=0) and P(w|c=1) is high -> P(x=1) is high
	        // if difference of P(w|c=0) and P(w|c=1) is low -> P(x=1) is low
	        
	        // collection-specific
	        // P(x=1|z,c) * P(z|w,c)
	        // ----- P(z|c) -----
	        p[1] = (nX[1][c][topic] + gamma1) * (nZWc[topic][w][c] + delta)
	                / (nZc[topic][c] + deltaNorm);
	        
	        pTotal = p[0] + p[1];
	
	        // sample u randomly from [0;pTotal)
	        u = r.nextDouble() * pTotal;
	
	        if (u > p[0])
	            route = 1; // collection-specific
	        else
	            route = 0; // collection-independent
        }
        // sample new value for topic
        // and incorporate topic distributions of neighbored documents 
        pTotal = 0.0;
        p = new double[Z];

        if (route == 0) {
            for (int z = 0; z < Z; z++) {
            	int did = d;
            	
            	p[z] = (nDZ[did][z] + myAlpha[c][z]) * (nZW[z][w] + beta)
                            / (nZ[z] + betaNorm);
                pTotal += p[z];
            }
        } else {
            for (int z = 0; z < Z; z++) {
            	int did = d;
            	
            	p[z] = (nDZ[did][z] + myAlpha[c][z]) * (nZWc[z][w][c] + delta)
                        / (nZc[z][c] + deltaNorm);
                pTotal += p[z];
            }
        }

        // sample u randomly from [0;pTotal)
        u = r.nextDouble() * pTotal;

        // iterate through topics until the cumulative probability of the topics is greater than u
        // "roulette wheel selection"
        double v = 0;
        for (int z = 0; z < Z; z++) {
            v += p[z];

            if (v > u) {
                topic = z;
                break;
            }
        }

        // increment counts
        nDZ[d][topic] += 1;
        nX[route][c][topic] += 1;

        if (route == 0) {
            nZW[topic][w] += 1;
            nZ[topic] += 1;
        } else {
            nZWc[topic][w][c] += 1;
            nZc[topic][c] += 1;
        }

        // set new assignments
        docsZ[d][n] = topic;
        docsX[d][n] = route;
    }
    
public void sampleTestSet(int d, int n, List<Integer> dids, ArrayList<ArrayList<Integer>> wordLists, ArrayList<Integer> doc2collection) {
    	
		int w = corpus.getWidTrain(wordLists.get(d).get(n));
        int c = doc2collection.get(d);
        int topic = docsZTestSet[d][n];
        int route;

        // decrement counts for document d's n-th word
        nDZTestSet[d][topic] -= 1;

        double betaNorm = W * beta;
        double deltaNorm = W * delta;

        // sample new value for route
        double pTotal = 0.0;
        double[] p = new double[2];
        double u;
        if(IncorporateWordFrequencies){
        	if(collectionSpecific[w]){
        		route = 1;
        	}
        	else{
        		route = 0;
        	}    
        }
        else{
        
	        p[0] = (nX[0][c][topic] + gamma0) * (nZW[topic][w] + beta)
	                / (nZ[topic] + betaNorm);
	        p[1] = (nX[1][c][topic] + gamma1) * (nZWc[topic][w][c] + delta)
	                / (nZc[topic][c] + deltaNorm);
	        
	        pTotal = p[0] + p[1];
	
	        // sample u randomly from [0;pTotal)
	        u = r.nextDouble() * pTotal;
	
	        if (u > p[0])
	            route = 1; // collection-specific
	        else
	            route = 0; // collection-independent
        }
        // sample new value for topic
        // and incorporate topic distributions of neighbored documents 
        pTotal = 0.0;
        p = new double[Z];

        // dids
        int did = d;
        if (route == 0) {
            for (int z = 0; z < Z; z++) {
            	p[z] = (nDZTestSet[did][z] + myAlpha[c][z]) * (nZW[z][w] + beta)
                            / (nZ[z] + betaNorm);
                pTotal += p[z];
            }
        } else {
            for (int z = 0; z < Z; z++) {
            	p[z] = (nDZTestSet[did][z] + myAlpha[c][z]) * (nZWc[z][w][c] + delta)
                        / (nZc[z][c] + deltaNorm);
                pTotal += p[z];
            }
        }

        // sample u randomly from [0;pTotal)
        u = r.nextDouble() * pTotal;

        // iterate through topics until the cumulative probability of the topics is greater than u
        // "roulette wheel selection"
        double v = 0;
        for (int z = 0; z < Z; z++) {
            v += p[z];

            if (v > u) {
                topic = z;
                break;
            }
        }

        // increment counts
        nDZTestSet[d][topic] += 1;

        // set new assignments
        docsZTestSet[d][n] = topic;
    }


public void sampleTestSetUnknownC(int d, int n, List<Integer> dids, ArrayList<ArrayList<Integer>> wordLists) {
  	// sampling for the test set if a document's collection is unknown
	  int w = corpus.getWidTrain(wordLists.get(d).get(n));
      int topic = docsZTestSet[d][n];
      int route;

      // decrement counts for document d's n-th word
      nDZTestSet[d][topic] -= 1;

      double betaNorm = W * beta;
      double deltaNorm = W * delta;

      // sample new value for route
      double pTotal = 0.0;
      double[] p = new double[2];
      double u;
      if(IncorporateWordFrequencies){
    	if(collectionSpecific[w]){
      		route = 1;
      	}
      	else{
      		route = 0;
      	}  
      }
      else{
      
    	  for(int c=0;c<C;c++){
	        p[0] += (nX[0][c][topic] + gamma0) * (nZW[topic][w] + beta)
	                / (nZ[topic] + betaNorm);
    	  }
    	  for(int c=0;c<C;c++){
	        p[1] = (nX[1][c][topic] + gamma1) * (nZWc[topic][w][c] + delta)
	                / (nZc[topic][c] + deltaNorm);
    	  }
	        pTotal += p[0] + p[1];
    	  
	        // sample u randomly from [0;pTotal)
	        u = r.nextDouble() * pTotal;
	
	        if (u > p[0])
	            route = 1; // collection-specific
	        else
	            route = 0; // collection-independent
      }
      // sample new value for topic
      // and incorporate topic distributions of neighbored documents 
      pTotal = 0.0;
      p = new double[Z];

      // dids
      int did = d;
      if (route == 0) {
          for (int z = 0; z < Z; z++) {
        	  for(int c=0;c<C;c++){
          	p[z] += (nDZTestSet[did][z] + myAlpha[c][z]) * (nZW[z][w] + beta)
                          / (nZ[z] + betaNorm);
        	  }
              pTotal += p[z];
          }
      } else {
          for (int z = 0; z < Z; z++) {
        	  for(int c=0;c<C;c++){
          	p[z] += (nDZTestSet[did][z] + myAlpha[c][z]) * (nZWc[z][w][c] + delta)
                      / (nZc[z][c] + deltaNorm);
        	  }
              pTotal += p[z];
          }
      }

      // sample u randomly from [0;pTotal)
      u = r.nextDouble() * pTotal;

      // iterate through topics until the cumulative probability of the topics is greater than u
      // "roulette wheel selection"
      double v = 0;
      for (int z = 0; z < Z; z++) {
          v += p[z];

          if (v > u) {
              topic = z;
              break;
          }
      }

      // increment counts
      nDZTestSet[d][topic] += 1;

      // set new assignments

      docsZTestSet[d][n] = topic;
  }

    private double[][] computePhi() {
        double[][] phi = new double[Z][W];

        for (int z = 0; z < Z; z++) {
            // Compute numerators and common denominator
            double sum = 0;
            for (int w = 0; w < W; w++) {
                phi[z][w] = nZW[z][w] + beta;
                sum += phi[z][w];
            }

            // Divide by common denominator
            for (int w = 0; w < W; w++) {
                phi[z][w] /= sum;
            }
        }
        return phi;
    }

    private double[][] computeSigmaForCollection(int c) {
        double[][] phi = new double[Z][W];
        for (int z = 0; z < Z; z++) {
            // Compute numerators and common denominator
            double sum = 0;
            for (int w = 0; w < W; w++) {
                phi[z][w] = nZWc[z][w][c] + beta; // beta is prior for topic-word distributions
                sum += phi[z][w];
            }

            // Divide by common denominator
            for (int w = 0; w < W; w++) {
                phi[z][w] /= sum;
            }
        }
        return phi;
    }
    
    private double[][] computeTheta(int[][] nDZ, ArrayList<ArrayList<Integer>> wordLists, ArrayList<Integer> doc2collection) {
    	int D = wordLists.size();
        double[][] theta = new double[D][Z];

        for (int d = 0; d < D; d++) {
            // Compute numerators and common denominator
        	
        	int c = doc2collection.get(d);
        			
            double sum = 0;
            for (int z = 0; z < Z; z++) {
                theta[d][z] = nDZ[d][z] + myAlpha[c][z];
                sum += theta[d][z];
            }

            // Divide by common denominator
            for (int z = 0; z < Z; z++) {
                theta[d][z] /= sum;
            }
        }

        return theta;
    }
}
