import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.Random;
import java.util.regex.Pattern;

import org.slf4j.Logger;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public abstract class Corpus {

	Logger logger;

	// helpers
	HashSet<String> stopwords;
	StanfordCoreNLP pipeline;
	HashMap<Integer, Integer> df;
	HashMap<String, Integer> w2widOld;
	HashMap<String, Integer> w2widNew;

	// configuration
	int removeStopWords;
	int removeRareWords;
	int removeCasing; // if true, text is converted to lower case
	int usePipeline;
	int useTfIdf;

	String name;

	// directories
	String baseDir;
	String outputDir;

	// matrices
	ArrayList<ArrayList<Integer>> did_widOld;
	ArrayList<ArrayList<Integer>> did_widNew;
	public ArrayList<ArrayList<Integer>> did_widNew_train;
	ArrayList<ArrayList<Integer>> did_widNew_test;

	public HashMap<Integer,Integer> wid2trainwid = new HashMap<Integer,Integer>();
	
	// dictionaries
	public ArrayList<String> wid2wNew;
	ArrayList<String> wid2wOld;
	public ArrayList<String> did2d;
	public HashMap<String,Integer> d2did;
	ArrayList<String> did_d_train;
	ArrayList<String> did_d_test;

	ArrayList<Integer> did2cid;
	ArrayList<ArrayList<Integer>> wid2count;
	ArrayList<Double> wid2proportions;
	public ArrayList<String> cid2c;
	ArrayList<Integer> did_c_train;
	ArrayList<Integer> did_c_test;
	
	//for cross-validation
	ArrayList<ArrayList<ArrayList<Integer>>> did_widNew_Sets;
	ArrayList<ArrayList<Integer>> did_c_Sets;
	ArrayList<ArrayList<String>> did_d_Sets;


	// statistics
	int numTypes;
	int numDocs;
	int C;
	public int Wtrain;

	double testPercentage = 0.1;
	double trainingPercentage = 0.9;
	
	Pattern bigramPattern = Pattern.compile("<phrase>([^< ]*) ([^< ]*)<\\/phrase>", Pattern.CASE_INSENSITIVE);
	Pattern trigramPattern = Pattern.compile("<phrase>([^< ]*) ([^< ]*) ([^< ]*)<\\/phrase>", Pattern.CASE_INSENSITIVE);
	Pattern quadgramPattern = Pattern.compile("<phrase>([^< ]*) ([^< ]*) ([^< ]*) ([^< ]*)<\\/phrase>", Pattern.CASE_INSENSITIVE);
	

	public ArrayList<ArrayList<Integer>> getTrainDocs() {

		return this.did_widNew_train;
	}
	

	public ArrayList<Integer> getTrainDocCollections() {

		return this.did_c_train;
	}

	public ArrayList<String> getTrainDocIds() {

		return this.did_d_train;
	}

	public ArrayList<ArrayList<Integer>> getTestDocs() {

		return this.did_widNew_test;
	}

	public ArrayList<Integer> getTestDocCollections() {

		return this.did_c_test;
	}

	public ArrayList<String> getTestDocIds() {

		return this.did_d_test;
	}

	abstract boolean createCorpus();


    public Corpus(Logger logger, String name, String baseDirectory, double corpusPercentage) {

        super();
        this.logger = logger;
        this.name = name;
        this.baseDir = baseDirectory;
        this.trainingPercentage = corpusPercentage;
        
    }


	public void init(int removeStopWords, int removeRareWords,
			int useTfIdf,int usePipeline) {
		init(removeStopWords, removeRareWords, useTfIdf, usePipeline, 1);
	}

	public void init(int removeStopWords, int removeRareWords,
			int useTfIdf,int usePipeline, int removeCasing) {

		this.logger.info("Corpus: initialize...");
		this.removeStopWords = removeStopWords;
		this.removeRareWords = removeRareWords;
		this.removeCasing = removeCasing;
		this.useTfIdf = useTfIdf;
		this.usePipeline = usePipeline;
		this.outputDir = this.baseDir+this.name+"-"+usePipeline+"-"+removeStopWords+"-"+removeRareWords+"-"+useTfIdf+"-"+removeCasing+"/";
		new File(this.outputDir).mkdirs();

		if (loadCorpus()) {
			this.logger.info("Loaded existing corpus with " + did2d.size() + " documents!");
		} else {
			initPreprocessing();
			createCorpus();
			postprocessing();
			storeCorpus();
			this.logger.info("Created corpus with " + did2d.size() + " documents!");
		}
	}

	private void postprocessing() {

		if (this.removeRareWords > 0) {
			Iterator<ArrayList<Integer>> iter = did_widOld.iterator();
			while(iter.hasNext()){
				ArrayList<Integer> wids = iter.next();
				ArrayList<Integer> widsNew = new ArrayList<Integer>();
				Iterator<Integer> iter2 = wids.iterator();
				while(iter2.hasNext()){
					Integer wid = iter2.next();
					if(this.df.get(wid)>this.removeRareWords){
						String wNew = this.wid2wOld.get(wid);
						Integer widNew = this.w2widNew.get(wNew);
						if(widNew==null){
							this.wid2wNew.add(wNew);
							widNew = this.w2widNew.size();
							this.w2widNew.put(wNew, widNew);
						}
						widsNew.add(widNew);
					}
				}
				this.did_widNew.add(widsNew);
			}
			this.wid2wOld = null;
			this.did_widOld = null;
		}else{
			this.wid2wNew = this.wid2wOld;
			this.did_widNew = this.did_widOld;
		}
		//clean up
		this.stopwords = null;
		this.pipeline = null;
		this.df = null;
		this.w2widOld = null;
		this.w2widNew = null;
	}

	private void initPreprocessing() {

		this.did2d = new ArrayList<String>();
		this.d2did = new HashMap<String,Integer>();
		this.did2cid = new ArrayList<Integer>();
		this.cid2c = new ArrayList<String>();
		this.wid2wOld = new ArrayList<String>();
		this.wid2wNew = new ArrayList<String>();
		this.did_widOld = new ArrayList<ArrayList<Integer>>();
		this.did_widNew = new ArrayList<ArrayList<Integer>>();
		this.w2widOld = new HashMap<String, Integer>();
		this.w2widNew = new HashMap<String, Integer>();
		if (this.removeRareWords > 0) {
			this.df = new HashMap<Integer, Integer>();
		}
		if (this.removeStopWords > 0) {
			this.stopwords = Utils.getStopwordList(Corpus.class.getResourceAsStream("/stop_words_eng_extended.txt"));
			this.stopwords.add("phrase");
			HashSet<String> germanStopwords = Utils.getStopwordList(Corpus.class.getResourceAsStream("/stop_words_de_extended.txt"));
			this.stopwords.addAll(germanStopwords);
			HashSet<String> frenchStopwords = Utils.getStopwordList(Corpus.class.getResourceAsStream("/stop_words_fr.txt"));
			this.stopwords.addAll(frenchStopwords);
		}
		
		if (this.usePipeline > 0) {
			Properties props = new Properties();
			props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
			
			this.pipeline = new StanfordCoreNLP(props);
		}
	}

	private void storeCorpus() {

		Utils.storeDict(this.did2d, this.outputDir + "did.doc.txt");
		Utils.storeDict(this.cid2c, this.outputDir + "cid.collection.txt");
		Utils.storeDict(this.did2cid, this.outputDir + "did_cid.txt");
		Utils.storeDict(this.wid2wNew, this.outputDir + "wid.word.txt");
		Utils.storeSparseMatrix(this.did_widNew, this.outputDir + "did_wid.txt");
		this.numTypes = this.wid2wNew.size();
		this.numDocs = this.did2d.size();
	}

	private boolean loadCorpus() {

		File wid2wFile = new File(this.outputDir + "wid.word.txt");
		File did2dFile = new File(this.outputDir + "did.doc.txt");
		File cid2cFile = new File(this.outputDir + "cid.collection.txt");
		File did2cidFile = new File(this.outputDir + "did_cid.txt");
		File did_widFile = new File(this.outputDir + "did_wid.txt");
		if (wid2wFile.exists() && did2dFile.exists() && did_widFile.exists()) {
			this.wid2wNew = Utils.loadDictionary(wid2wFile);
			this.did2d = Utils.loadDictionary(did2dFile);
			this.cid2c = Utils.loadDictionary(cid2cFile);
			this.did2cid = Utils.loadIntegerDictionary(did2cidFile);
			this.did_widNew = Utils.loadMatrix(did_widFile);
			this.numTypes = this.wid2wNew.size();
			this.numDocs = this.did2d.size();
			this.d2did = new HashMap<String,Integer>();
			for(int i=0;i<this.did2d.size();i++){
				String docName = this.did2d.get(i);
				this.d2did.put(docName, i);	
			}
			return true;
		}
		return false;
	}

	boolean addDocToCorpus(String docName, String docText) {
        return addDocToCorpus(docName, docText, "0"); // default collection "0"
    }

    void addCollectionToDocument(String collection){
    	Integer cid = this.cid2c.indexOf(collection); // potentially slow, HashMap can make this faster
		if (cid == -1) {
			cid = this.cid2c.size();
			this.cid2c.add(collection);
		}
		did2cid.add(cid);
    }
 

    boolean addDocToCorpus(String docName, String docText, String collection) {

		if (this.did2d.contains(docName)) {
			this.logger.info("Doc already in Corpus! " + docName);
			return false;
		}
		ArrayList<Integer> doc = new ArrayList<Integer>();
		ArrayList<String> tokens = preprocess(docText, removeStopWords, usePipeline, removeCasing);
		Iterator<String> iter = tokens.iterator();
		while (iter.hasNext()) {
			String word = iter.next();
			Integer wid = this.w2widOld.get(word);
			if (wid == null) {
				this.wid2wOld.add(word);
				wid = this.w2widOld.size();
				this.w2widOld.put(word, wid);
			}
			doc.add(wid);
		}
		//if (doc.size() < 3) {
		//	this.logger.info("Doc less than 3 words! " + docName);
		//	return false;
		//}
		this.did_widOld.add(doc);
		this.did2d.add(docName);
		this.d2did.put(docName, this.did2d.size()-1);
		addCollectionToDocument(collection);
		if (this.removeRareWords > 0) {
			HashSet<Integer> set = new HashSet<Integer>(doc);
			Iterator<Integer> iter2 = set.iterator();
			while (iter2.hasNext()) {
				Integer wid2 = iter2.next();
				Integer oldCount = this.df.get(wid2);
				if (oldCount == null)
					oldCount = 0;
				this.df.put(wid2, ++oldCount);
			}
		}
		return true;
	}
    private  ArrayList<String> preprocess(String docText, int removeStopWords, int usePipeline) {
    	return  preprocess(docText, removeStopWords, usePipeline, 1);
    }
	private  ArrayList<String> preprocess(String docText, int removeStopWords, int usePipeline, int removeCasing) {
		if(1 == removeCasing){
			docText = docText.toLowerCase();
		}
		boolean autoPhrase=true;
		if(autoPhrase){
			docText = bigramPattern.matcher(docText).replaceAll("$1_$2");
			docText = trigramPattern.matcher(docText).replaceAll("$1_$2_$3");
			docText = quadgramPattern.matcher(docText).replaceAll("$1_$2_$3_$4");
		}
		

		ArrayList<String> filtered = new ArrayList<String>();
		if (usePipeline > 0) {
			Annotation annotation = this.pipeline.process(docText);
			List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
			for (CoreMap sentence : sentences) {
				for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
					String pos = token.get(PartOfSpeechAnnotation.class);
					if (pos.startsWith("-"))
						continue;
					String term = token.get(LemmaAnnotation.class);
					term = term.replaceAll("[^a-zA-ZßüäöÜÄÖ_]", ""); 
					if (term.length() < 3 || term.length() > 40)
						continue;
					if (removeStopWords > 0 && this.stopwords.contains(term))
						continue;
					filtered.add(term);
				}
			}
		} else {
			String text = docText.replaceAll("\n", " ");
			text = text.replaceAll("- *", "");
			text = text.replaceAll("\\p{Alpha}", " ");
			text = text.toLowerCase();
			String[] textArray = text.split("\\p{Space}");
			for (int i = 0; i < textArray.length; i++) {
				String term = textArray[i].trim();
				if (term.length() < 3 || term.length() > 40)
					continue;
				if (removeStopWords > 0 && this.stopwords.contains(term))
					continue;
				filtered.add(term);
			}
		}
		return filtered;
	}
	
	public void initializeForCrossValidationRun(int indexOfTestSet){
		//indexOfTestSet = -1; //TODO full dataset
		int numberOfSets = did_widNew_Sets.size();
		this.did_widNew_test = did_widNew_Sets.get(indexOfTestSet);
		this.did_widNew_train = new ArrayList<ArrayList<Integer>>();
		for(int i=0;i<numberOfSets;i++){
			//if(i!=indexOfTestSet){
				this.did_widNew_train.addAll(did_widNew_Sets.get(i));
			//}
		}
		
		this.did_c_test = did_c_Sets.get(indexOfTestSet);
		this.did_c_train = new ArrayList<Integer>();
		for(int i=0;i<numberOfSets;i++){
			//if(i!=indexOfTestSet){
				this.did_c_train.addAll(did_c_Sets.get(i));
			//}
		}
		
		this.did_d_test = did_d_Sets.get(indexOfTestSet);
		this.did_d_train = new ArrayList<String>();
		for(int i=0;i<numberOfSets;i++){
			//if(i!=indexOfTestSet){
				this.did_d_train.addAll(did_d_Sets.get(i));
			//}
		}
		
		countOccurrences();
	}
	
	public void splitForCrossValidation(int numberOfSets) {
//		int searchId =0;
//		for(int i=0;i<wid2wNew.size();i++){
//			String word = wid2wNew.get(i);
//			if(word.equalsIgnoreCase("vorurteile")){
//				searchId=i;
//			}
//		}
//		int countOcC0=0;
//		int countOcC1=0;
//		int did=0;
//		for(ArrayList<Integer> wid : did_widNew){
//			if(did2cid.get(did)==0){
//			for(Integer w : wid){
//				if(w==searchId){
//					countOcC0++;
//				}
//			}
//			}else{
//				for(Integer w : wid){
//					if(w==searchId){
//						countOcC1++;
//					}
//				}
//			}
//			did++;
//		}
//		logger.info("counted "+countOcC0);
//		logger.info("counted "+countOcC1);
//		
		int numberOfDocuments = did_widNew.size();
		for(Integer c : did2cid){
			if(c>C){
				C=c;
			}
		}
		C++;
		int numberOfDocumentsPerSet = numberOfDocuments/numberOfSets;

		Collections.shuffle(did_widNew, new Random(1464084109885L));
		did_widNew_Sets = new ArrayList<ArrayList<ArrayList<Integer>>>(numberOfSets);
		for(int i=0; i< numberOfSets; i++){
			did_widNew_Sets.add(new ArrayList<ArrayList<Integer>>(did_widNew.subList(i*numberOfDocumentsPerSet, (i+1)*numberOfDocumentsPerSet)));
		}

		Collections.shuffle(did2cid, new Random(1464084109885L));
		did_c_Sets = new ArrayList<ArrayList<Integer>>(numberOfSets);
		for(int i=0; i< numberOfSets; i++){
			did_c_Sets.add(new ArrayList<Integer>(did2cid.subList(i*numberOfDocumentsPerSet, (i+1)*numberOfDocumentsPerSet)));
		}		
		
		Collections.shuffle(did2d, new Random(1464084109885L));
		did_d_Sets = new ArrayList<ArrayList<String>>(numberOfSets);
		for(int i=0; i< numberOfSets; i++){
			did_d_Sets.add(new ArrayList<String>(did2d.subList(i*numberOfDocumentsPerSet, (i+1)*numberOfDocumentsPerSet)));
		}
		// all collections (did_widNew, did2cid, did2d) are shuffled and split the same way by using the same seed
	}
	public int getOccurrence(int wid, int cid){
		return wid2count.get(cid).get(wid)+1;
	}
	public int getTotalOccurrence(int wid){
		int sum=0;
		for(int c=0;c<C;c++){
			sum+=getOccurrence(wid, c);
		}
		return sum;
	}
	public int getWidTrain(int wid){
		Integer mappedWid = wid2trainwid.get(wid);
		if(mappedWid==null){
			return -1;
		}
		return mappedWid;
	}
	
	public void countOccurrences(){
		wid2trainwid = new HashMap<Integer,Integer>();
		wid2count = new ArrayList<ArrayList<Integer>>();	
		
		int numberOfWords=0;
		for(ArrayList<Integer> wids : did_widNew_train){
			for(Integer wid : wids){
				if(!wid2trainwid.containsKey(wid)){
					wid2trainwid.put(wid, numberOfWords);
					numberOfWords++;
				}
			}
		}
		Wtrain = wid2trainwid.size();
		for(int cid = 0; cid<C;cid++){
			ArrayList<Integer> occurrences = new ArrayList<Integer>();
			for(int wid=0; wid<Wtrain;wid++){
				occurrences.add(0);
			}
			wid2count.add(occurrences);
		}
		int did =0;
		for(ArrayList<Integer> wids : did_widNew_train){
			int cid = did_c_train.get(did);
			for(Integer wid : wids){
				int previousCount = wid2count.get(cid).get(wid2trainwid.get(wid));
				wid2count.get(cid).set(wid2trainwid.get(wid), previousCount+1);
			}
			did++;
		}
	}
}
