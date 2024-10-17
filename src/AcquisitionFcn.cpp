#define ppi 3.1415926535354

#include "AcquisitionFcn.h"

AcquisitionFcn::AcquisitionFcn()
	: acquisition_func("probability_of_improvement") {

}

AcquisitionFcn::AcquisitionFcn(std::string afcn_name)
	: acquisition_func(afcn_name) {

}

Eigen::VectorXd AcquisitionFcn::getNextCandiatePoint() const {
	return candiate_predictor;
}

AFcnPI AcquisitionFcn::fitProbabilityOfImprovement(const Eigen::MatrixXd domainX, objectiveFcnGP obj, 
	const double FBest, const double margin) {
	Eigen::RowVectorXd PI(domainX.cols());
	Eigen::RowVectorXd FSD(domainX.cols());
	Eigen::RowVectorXd GammaX(domainX.cols());
	return { PI, FSD, GammaX };
}

AFcnEI AcquisitionFcn::fitExpectedImprovement(const Eigen::MatrixXd domainX, objectiveFcnGP obj, 
	double FBest) {
	AFcnPI afcnPI = fitProbabilityOfImprovement(domainX, obj, FBest, 0);
	Eigen::RowVectorXd EI = afcnPI.FSD.array() * 
		(afcnPI.GammaX.array() * afcnPI.PI.array() + normPDF(afcnPI.GammaX, 0, 1).array());

	// % If PI==0, set EI=0???
	EI = (afcnPI.PI.array() == 0).select(0, EI);

	return { EI };
}

Eigen::RowVectorXd normPDF(const Eigen::RowVectorXd& X, const double norm_mu, const double norm_sigma) {
	Eigen::RowVectorXd exponent = -0.5 * ((X.array() - norm_mu) / norm_sigma).array().square();
	return exponent.array().exp() / (std::sqrt(2 * ppi) * norm_sigma);
}