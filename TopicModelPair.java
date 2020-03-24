public class TopicModelPair {
	public double[][] phi;
	public double[][][] sigmaC;
	
	public TopicModelPair(double[][] phi, double[][][] phiC) {
		this.phi = phi;
		this.sigmaC = phiC;
	}
	public double[][] getFirst(){
		return phi;
	}
	public double[][][] getSecond(){
		return sigmaC;
	}
}
