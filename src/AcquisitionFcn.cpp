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

AFcnPI AcquisitionFcn::fitProbabilityOfImprovement(const double FBest, const double margin) {

}