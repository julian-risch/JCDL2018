import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.action.search.SearchType;
import org.elasticsearch.client.transport.TransportClient;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.transport.InetSocketTransportAddress;
import org.elasticsearch.common.unit.TimeValue;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.index.query.RangeQueryBuilder;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.sort.SortOrder;
import org.slf4j.Logger;

public class CorpusBlogPostsFromFile extends Corpus {
	
	public CorpusBlogPostsFromFile(Logger logger, String name, String baseDirectory, double trainingPercentage) {
		super(logger, CorpusBlogPostsFromFile.name, baseDirectory, trainingPercentage);
	}

	public CorpusBlogPostsFromFile(Logger logger, String baseDirectory, double trainingPercentage) {
		super(logger, CorpusBlogPostsFromFile.name, baseDirectory, trainingPercentage);
	}

	final static String name = "blogposts";

	@Override
	boolean createCorpus() {
		try {
			return readFile();
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
	}
	boolean readFile() throws IOException{
		FileInputStream fstream = new FileInputStream(Corpus.class.getResource("/blogpostsnew/segmentation.txt").getPath());
		FileInputStream idStream = new FileInputStream(Corpus.class.getResource("/blogpostsnew/blogposts-id.txt").getPath()); 
		BufferedReader br = new BufferedReader(new InputStreamReader(fstream));
		BufferedReader idBr = new BufferedReader(new InputStreamReader(idStream));
		String strLine;

		//Read File Line By Line
		Pattern idPattern = Pattern.compile("<id>[^<]*</id>", Pattern.CASE_INSENSITIVE);
		Pattern collectionPattern = Pattern.compile("<collection>[^<]*</collection>", Pattern.CASE_INSENSITIVE);
		
		while ((strLine = br.readLine()) != null)   {
			String idLine = idBr.readLine();
			Matcher idMatcher = idPattern.matcher(idLine);
			Matcher collectionMatcher = collectionPattern.matcher(idLine);
			idMatcher.find();
			collectionMatcher.find();
			String docName="";
			try{
			docName = idMatcher.group(0);
			}
			catch(Exception e){
				System.out.println(strLine);
				continue;
			}
			docName = docName.substring(4,docName.indexOf("</id>"));
			
			String country = collectionMatcher.group(0);
			country = country.substring(12,country.indexOf("</collection>"));
			
			String docText = strLine;
			this.addDocToCorpus(docName, docText, country);
		}

		//Close the input stream
		br.close();
		idBr.close();
		return true;
	}

}
