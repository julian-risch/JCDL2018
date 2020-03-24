import org.slf4j.Logger;

public class CorpusToy extends Corpus {

	final String inputDir = null;
	final static String name = "ldaToy";


	String[][] docs = {
        {"d1", "bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream", "1993", "A"},
        {"d2", "bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money", "1994", "B"},
        {"d3", "bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream", "1993", "A"},
        {"d4", "bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money", "1994", "B"},
        {"d5", "bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream", "1993", "A"},
        {"d6", "bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money", "1994", "B"},
        {"d7", "bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream", "1993", "A"},
        {"d8", "bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money", "1994", "B"},
        {"d9", "bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream bank river stream", "1993", "A"},
        {"d10", "bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money bank loan money", "1994", "B"},
	};

	public CorpusToy(Logger logger, String baseDirectory, double trainingPercentage) {
		super(logger, CorpusToy.name, baseDirectory, trainingPercentage);
		this.testPercentage = 0.5;
	}

	@Override
	boolean createCorpus() {

		for(int i=0; i<this.docs.length;i++){
			this.addDocToCorpus(this.docs[i][0], this.docs[i][1], this.docs[i][3]);
		}
		if(this.docs.length>0) return true;
		else return false;
	}
}
