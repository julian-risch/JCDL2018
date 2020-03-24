import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map.Entry;
import java.util.TreeMap;

import org.slf4j.Logger;

public abstract class TopicModel {

	protected Logger logger;
	String name;
	protected Corpus corpus;
	protected int topicNumber;

	// directories
	protected String outputDir;

	// topic model
	protected double[][] phi;
	protected double [][] theta;

	public TopicModel(Logger logger, String name, Corpus corpus) {
		super();
		this.logger = logger;
		this.name = name;
		this.corpus = corpus;
	}

	public void init(int topicNumber){

		this.logger.info("TopicModel: initialize...");
		this.outputDir = this.corpus.outputDir +name+"-"+topicNumber +"/";
		this.topicNumber = topicNumber;
		new File(this.outputDir).mkdirs();
	}

	public abstract void train();

    public double test() { return 0; }

	public void storeModel() {

		Utils.storeMatrix(this.phi, this.outputDir + "phi.txt");
		Utils.storeMatrix(this.theta, this.outputDir + "theta.txt");
		Utils.storeMapping(this.getTopTopicTerms(this.phi), outputDir +"tid_terms.txt");
	}



	public TreeMap<String,Integer> getTopTopicTerms(double[][] phi){

		TreeMap<Integer, LinkedHashMap<Integer, Double>> sortedMap = Utils.getSortedTreeMap(phi);
		TreeMap<String,Integer> res = new TreeMap<String,Integer>();
		Iterator<Entry<Integer, LinkedHashMap<Integer, Double>>> iter = sortedMap.entrySet().iterator();
		while(iter.hasNext()){
			Entry<Integer, LinkedHashMap<Integer, Double>> entry = iter.next();
			Integer tid = entry.getKey();
			String terms = "";
			int count = 0;
			Iterator<Integer> iter2 = entry.getValue().keySet().iterator();
			while(iter2.hasNext() && count++ < 30){
				int wid = iter2.next();
				terms += this.corpus.wid2wNew.get(wid) +" ";
			}
			res.put(terms, tid);
		}
		return res;
	}

	/**
	 * Should return an M x K array (M = number of test documents) where
	 * theta[m][k] = proportion of topic k in test document m.
	 * @return
	 */
	public abstract double[][] getTestSetTheta();

	/**
	 * Log Likelihood of the Test Set given this.phi and theta = getTestSetTheta()
	 * @return the log likelihood
	 */
	public double testSetLogLikelihood() {
		double[][] theta = getTestSetTheta();
		ArrayList<ArrayList<Integer>> wordLists = corpus.getTestDocs();
		return computeLogLikelihood(wordLists, phi, theta);
	}

	/**
	 * Log Likelihood of the Test Set given this.phi and theta = getTestSetTheta()
	 * @return the log likelihood
	 */
	public double computeLogLikelihood(ArrayList<ArrayList<Integer>> wordLists, double[][] phi, double[][] theta) {
		int M = wordLists.size();
		double logLikelihood = 0;

		for (int m = 0; m < M; m++) {
			for (int t : wordLists.get(m)) {
				double wordLikelihood = 0;
				for (int k = 0; k < topicNumber; k++) {
					wordLikelihood += theta[m][k] * phi[k][t];
				}
				logLikelihood += Math.log(wordLikelihood);
			}
		}

		return logLikelihood;
	}


	/**
	 * Computed from log likelihood, see
	 * https://en.wikipedia.org/wiki/Perplexity#Perplexity_of_a_probability_model
	 */
	public double testSetPerplexity() {
      double logLikelihood = testSetLogLikelihood();
	  return computeSetPerplexity(logLikelihood, corpus.getTestDocs());
	}

	/**
	 * Computed from log likelihood, see
	 * https://en.wikipedia.org/wiki/Perplexity#Perplexity_of_a_probability_model
	 */
	public double computeSetPerplexity(double logLikelihood, ArrayList<ArrayList<Integer>> wordLists) {
		int N = 0;
		for (ArrayList<Integer> wordList : wordLists) {
			N += wordList.size();
		}

		return Math.exp(- 1.0 / N * logLikelihood);
	}

}
