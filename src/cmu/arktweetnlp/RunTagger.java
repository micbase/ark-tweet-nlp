package cmu.arktweetnlp;

//package py4j.examples;
import py4j.GatewayServer;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.LineNumberReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.HashSet;
import java.util.List;

import cmu.arktweetnlp.impl.ModelSentence;
import cmu.arktweetnlp.impl.Sentence;
import cmu.arktweetnlp.impl.features.FeatureExtractor;
import cmu.arktweetnlp.impl.features.WordClusterPaths;
import cmu.arktweetnlp.io.CoNLLReader;
import cmu.arktweetnlp.io.JsonTweetReader;
import cmu.arktweetnlp.util.BasicFileIO;
import edu.stanford.nlp.util.StringUtils;

import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;

/**
 * Commandline interface to run the Twitter POS tagger with a variety of possible input and output formats.
 * Also does basic evaluation if given labeled input text.
 * 
 * For basic usage of the tagger from Java, see instead Tagger.java.
 */
public class RunTagger {
	Tagger tagger;
	
	// Commandline I/O-ish options
	String inputFormat = "text";
	String outputFormat = "conll";
	int inputField = 1;
	
	String inputFilename;
	/** Can be either filename or resource name **/
	String modelFilename = "/cmu/arktweetnlp/model.20120919";

	public boolean noOutput = false;
	public boolean justTokenize = false;
	
	public static enum Decoder { GREEDY, VITERBI };
	public Decoder decoder = Decoder.GREEDY; 
	public boolean showConfidence = false;

	PrintStream outputStream;
	Iterable<Sentence> inputIterable = null;
	
	// Evaluation stuff
	private static HashSet<String> _wordsInCluster;
	// Only for evaluation mode (conll inputs)
	int numTokensCorrect = 0;
	int numTokens = 0;
	int oovTokensCorrect = 0;
	int oovTokens = 0;
	int clusterTokensCorrect = 0;
	int clusterTokens = 0;
	
	public static void die(String message) {
		// (BTO) I like "assert false" but assertions are disabled by default in java
		System.err.println(message);
		System.exit(-1);
	}

    //public RunTagger() {
    //}

	public RunTagger() throws IOException, UnsupportedEncodingException {
		// force UTF-8 here, so don't need -Dfile.encoding
        tagger = new Tagger();
        if (!justTokenize) {
            tagger.loadModel(modelFilename);			
            System.err.println("Model Loaded.\n");
        }
		this.outputStream = new PrintStream(System.out, true, "UTF-8");
	}
	public void detectAndSetInputFormat(String tweetData) throws IOException {
		JsonTweetReader jsonTweetReader = new JsonTweetReader();
		if (jsonTweetReader.isJson(tweetData)) {
			System.err.println("Detected JSON input format");
			inputFormat = "json";
		} else {
			System.err.println("Detected text input format");
			inputFormat = "text";
		}
	}

    public String voting_with4(double s1, double s2, String s3, String s4, String s5) {

        FastVector      atts;
        FastVector      attVals;
        Instance        inst;
        double[]        vals;
        int             i;
        Instances data;

        atts = new FastVector();
        atts.addElement(new Attribute("s1"));
        atts.addElement(new Attribute("s2"));
        attVals = new FastVector();
        attVals.addElement("P");
        attVals.addElement("N");
        attVals.addElement("O");
        atts.addElement(new Attribute("s3", attVals));
        atts.addElement(new Attribute("s4", attVals));
        atts.addElement(new Attribute("s5", attVals));

        vals = new double[5];
        vals[0] = s1;
        vals[1] = s2;
        vals[2] = attVals.indexOf(s3);
        vals[3] = attVals.indexOf(s4);
        vals[4] = attVals.indexOf(s5);
        inst = new Instance(1.0, vals);

        data = new Instances("Sentiment", atts, 0);
        data.add(inst);
        data.setClassIndex(4);

        try {
            Classifier tree = (Classifier) weka.core.SerializationHelper.read("./sentiment_with4.model");
            double clsLabel = tree.classifyInstance(data.instance(0));
            return data.classAttribute().value((int) clsLabel);
        }
        catch (FileNotFoundException e){
            System.out.println(e);
            return "Error file not found";
        }
        catch (IOException e){
            System.out.println(e);
            return "Error io";
        }
        catch (Exception e){
            System.out.println(e);
            return "Error";
        }
    }
    
    public String voting_without4(double s1, double s2, String s3, String s5) {

        FastVector      atts;
        FastVector      attVals;
        Instance        inst;
        double[]        vals;
        int             i;
        Instances data;

        atts = new FastVector();
        atts.addElement(new Attribute("s1"));
        atts.addElement(new Attribute("s2"));
        attVals = new FastVector();
        attVals.addElement("P");
        attVals.addElement("N");
        attVals.addElement("O");
        atts.addElement(new Attribute("s3", attVals));
        atts.addElement(new Attribute("s5", attVals));

        vals = new double[4];
        vals[0] = s1;
        vals[1] = s2;
        vals[2] = attVals.indexOf(s3);
        vals[3] = attVals.indexOf(s5);
        inst = new Instance(1.0, vals);

        data = new Instances("Sentiment", atts, 0);
        data.add(inst);
        data.setClassIndex(3);

        try {
            Classifier tree = (Classifier) weka.core.SerializationHelper.read("./sentiment_without4.model");
            double clsLabel = tree.classifyInstance(data.instance(0));
            return data.classAttribute().value((int) clsLabel);
        }
        catch (FileNotFoundException e){
            System.out.println(e);
            return "Error file not found";
        }
        catch (IOException e){
            System.out.println(e);
            return "Error io";
        }
        catch (Exception e){
            System.out.println(e);
            return "Error";
        }
    }

    public String test() {
        return "TESTOK";
    }

	public String runTagger(String text) throws IOException, ClassNotFoundException {
		
        Sentence sentence = new Sentence();

        sentence.tokens = Twokenize.tokenizeRawTweetText(text);
        ModelSentence modelSentence = null;

        if (sentence.T() > 0 && !justTokenize) {
            modelSentence = new ModelSentence(sentence.T());
            tagger.featureExtractor.computeFeatures(sentence, modelSentence);
            goDecode(modelSentence);
        }

        return outputJustTagging(sentence, modelSentence);
	}

	/** Runs the correct algorithm (make config option perhaps) **/
	public void goDecode(ModelSentence mSent) {
		if (decoder == Decoder.GREEDY) {
			tagger.model.greedyDecode(mSent, showConfidence);
		} else if (decoder == Decoder.VITERBI) {
//			if (showConfidence) throw new RuntimeException("--confidence only works with greedy decoder right now, sorry, yes this is a lame limitation");
			tagger.model.viterbiDecode(mSent);
		}		
	}
	
	public void runTaggerInEvalMode() throws IOException, ClassNotFoundException {
		
		long t0 = System.currentTimeMillis();
		int n=0;

		List<Sentence> examples = CoNLLReader.readFile(inputFilename); 
		inputIterable = examples;

		int[][] confusion = new int[tagger.model.numLabels][tagger.model.numLabels];
		
		for (Sentence sentence : examples) {
			n++;
			
			ModelSentence mSent = new ModelSentence(sentence.T());
			tagger.featureExtractor.computeFeatures(sentence, mSent);
			goDecode(mSent);
			
			if ( ! noOutput) {
				outputJustTagging(sentence, mSent);	
			}
			evaluateSentenceTagging(sentence, mSent);
			//evaluateOOV(sentence, mSent);
			//getconfusion(sentence, mSent, confusion);
		}

		System.err.printf("%d / %d correct = %.4f acc, %.4f err\n", 
				numTokensCorrect, numTokens,
				numTokensCorrect*1.0 / numTokens,
				1 - (numTokensCorrect*1.0 / numTokens)
		);
		double elapsed = ((double) (System.currentTimeMillis() - t0)) / 1000.0;
		System.err.printf("%d tweets in %.1f seconds, %.1f tweets/sec\n",
				n, elapsed, n*1.0/elapsed);
		
/*		System.err.printf("%d / %d cluster words correct = %.4f acc, %.4f err\n", 
				oovTokensCorrect, oovTokens,
				oovTokensCorrect*1.0 / oovTokens,
				1 - (oovTokensCorrect*1.0 / oovTokens)
		);	*/
/*		int i=0;
		System.out.println("\t"+tagger.model.labelVocab.toString().replaceAll(" ", ", "));
		for (int[] row:confusion){
			System.out.println(tagger.model.labelVocab.name(i)+"\t"+Arrays.toString(row));
			i++;
		}		*/
	}
	
	private void evaluateOOV(Sentence lSent, ModelSentence mSent) throws FileNotFoundException, IOException, ClassNotFoundException {
		for (int t=0; t < mSent.T; t++) {
			int trueLabel = tagger.model.labelVocab.num(lSent.labels.get(t));
			int predLabel = mSent.labels[t];
			if(wordsInCluster().contains(lSent.tokens.get(t))){
				oovTokensCorrect += (trueLabel == predLabel) ? 1 : 0;
				oovTokens += 1;
			}
		}
    }
	private void getconfusion(Sentence lSent, ModelSentence mSent, int[][] confusion) {
		for (int t=0; t < mSent.T; t++) {
			int trueLabel = tagger.model.labelVocab.num(lSent.labels.get(t));
			int predLabel = mSent.labels[t];
			if(trueLabel!=-1)
				confusion[trueLabel][predLabel]++;
		}
		
		
    }
	public void evaluateSentenceTagging(Sentence lSent, ModelSentence mSent) {
		for (int t=0; t < mSent.T; t++) {
			int trueLabel = tagger.model.labelVocab.num(lSent.labels.get(t));
			int predLabel = mSent.labels[t];
			numTokensCorrect += (trueLabel == predLabel) ? 1 : 0;
			numTokens += 1;
		}
	}
	
	private String formatConfidence(double confidence) {
		// too many decimal places wastes space
		return String.format("%.4f", confidence);
	}

	/**
	 * assume mSent's labels hold the tagging.
	 */
	public String outputJustTagging(Sentence lSent, ModelSentence mSent) {
		// mSent might be null!
        String result = "[";

        for (int t=0; t < lSent.T(); t++) {
            if (t < lSent.T() - 1)
                result += String.format("{\"word\":\"%s\", \"pos\":\"%s\"},",
                        lSent.tokens.get(t),  
                        tagger.model.labelVocab.name(mSent.labels[t]));
            else
                result += String.format("{\"word\":\"%s\", \"pos\":\"%s\"}",
                        lSent.tokens.get(t),  
                        tagger.model.labelVocab.name(mSent.labels[t]));

        }
        result += "]";

        return result;
	}
	/**
	 * assume mSent's labels hold the tagging.
	 * 
	 * @param lSent
	 * @param mSent
	 * @param inputLine -- assume does NOT have trailing newline.  (default from java's readLine)
	 */
	public void outputPrependedTagging(Sentence lSent, ModelSentence mSent, 
			boolean suppressTags, String inputLine) {
		// mSent might be null!
		
		int T = lSent.T();
		String[] tokens = new String[T];
		String[] tags = new String[T];
		String[] confs = new String[T];
		for (int t=0; t < T; t++) {
			tokens[t] = lSent.tokens.get(t);
			if (!suppressTags) {
				tags[t] = tagger.model.labelVocab.name(mSent.labels[t]);	
			}
			if (showConfidence) {
				confs[t] = formatConfidence(mSent.confidences[t]);
			}
		}
		
		StringBuilder sb = new StringBuilder();
		sb.append(StringUtils.join(tokens));
		sb.append("\t");
		if (!suppressTags) {
			sb.append(StringUtils.join(tags));
			sb.append("\t");
		}
		if (showConfidence) {
			sb.append(StringUtils.join(confs));
			sb.append("\t");
		}
		sb.append(inputLine);
		
		outputStream.println(sb.toString());
	}


	///////////////////////////////////////////////////////////////////


	public static void main(String[] args) throws IOException, ClassNotFoundException {        
		RunTagger tagger = new RunTagger();
		tagger.finalizeOptions();

        int port = 25333;
        if (args.length == 1)
            port = Integer.parseInt(args[0]);

        GatewayServer gatewayServer = new GatewayServer(tagger, port);
        gatewayServer.start();
        System.out.println("Gateway Server Started");

        //tagger.runTagger(args[0]);		
	}
	
	public void finalizeOptions() throws IOException {
		if (outputFormat.equals("auto")) {
			if (inputFormat.equals("conll")) {
				outputFormat = "conll";
			} else {
				outputFormat = "pretsv";
			}
		}
		if (showConfidence && decoder==Decoder.VITERBI) {
			System.err.println("Confidence output is unimplemented in Viterbi, turning it off.");
			showConfidence = false;
		}
		if (justTokenize) {
			showConfidence = false;
		}
	}
	
	public static void usage() {
		usage(null);
	}

	public static void usage(String extra) {
		System.out.println(
"RunTagger [options] [ExamplesFilename]" +
"\n  runs the CMU ARK Twitter tagger on tweets from ExamplesFilename, " +
"\n  writing taggings to standard output. Listens on stdin if no input filename." +
"\n\nOptions:" +
"\n  --model <Filename>        Specify model filename. (Else use built-in.)" +
"\n  --just-tokenize           Only run the tokenizer; no POS tags." +
"\n  --quiet                   Quiet: no output" +
"\n  --input-format <Format>   Default: auto" +
"\n                            Options: json, text, conll" +
"\n  --output-format <Format>  Default: automatically decide from input format." +
"\n                            Options: pretsv, conll" +
"\n  --input-field NUM         Default: 1" +
"\n                            Which tab-separated field contains the input" +
"\n                            (1-indexed, like unix 'cut')" +
"\n                            Only for {json, text} input formats." +
"\n  --word-clusters <File>    Alternate word clusters file (see FeatureExtractor)" +
"\n  --no-confidence           Don't output confidence probabilities" +
"\n  --decoder <Decoder>       Change the decoding algorithm (default: greedy)" +
"\n" +
"\nTweet-per-line input formats:" +
"\n   json: Every input line has a JSON object containing the tweet," +
"\n         as per the Streaming API. (The 'text' field is used.)" +
"\n   text: Every input line has the text for one tweet." +
"\nWe actually assume input lines are TSV and the tweet data is one field."+
"\n(Therefore tab characters are not allowed in tweets." +
"\nTwitter's own JSON formats guarantee this;" +
"\nif you extract the text yourself, you must remove tabs and newlines.)" +
"\nTweet-per-line output format is" +
"\n   pretsv: Prepend the tokenization and tagging as new TSV fields, " +
"\n           so the output includes a complete copy of the input." +
"\nBy default, three TSV fields are prepended:" +
"\n   Tokenization \\t POSTags \\t Confidences \\t (original data...)" +
"\nThe tokenization and tags are parallel space-separated lists." +
"\nThe 'conll' format is token-per-line, blank spaces separating tweets."+
"\n");
		
		if (extra != null) {
			System.out.println("ERROR: " + extra);
		}
		System.exit(1);
	}
	public static HashSet<String> wordsInCluster() {
		if (_wordsInCluster==null) {
			_wordsInCluster = new HashSet<String>(WordClusterPaths.wordToPath.keySet());
		}
		return _wordsInCluster;
	}
}
