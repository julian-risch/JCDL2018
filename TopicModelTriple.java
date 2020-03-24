public class TopicModelTriple {
    public TopicModelTriple(double[][] phi, double[][][] phiC, double[][] theta) {
		this.phi = phi;
		this.sigmaC = phiC;
		this.theta = theta;
	}
	public double[][] phi;
    public double[][][] sigmaC;        
    public double[][] theta;
    public double[][] getLeft(){
    	return phi;
    }
    public double[][][] getMiddle(){
    	return sigmaC;
    }
	public double[][] getRight(){
		return theta;
	}
}
