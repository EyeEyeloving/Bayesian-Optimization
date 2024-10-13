#include "AcquisitionFunc.h"

AcquisitionFunc::AcquisitionFunc()
	: acquisition_func("probability_of_improvement") {

}

AcquisitionFunc::AcquisitionFunc(std::string afcn_name)
	: acquisition_func(afcn_name) {

}

AFcnPI AcquisitionFunc::fitProbabilityOfImprovement(const double FBest, const double margin) {

}