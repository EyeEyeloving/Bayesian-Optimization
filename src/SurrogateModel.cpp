#include "SurrogateModel.h"
#include "GaussianProcess.h"

SurrogateModel::SurrogateModel() {

}

void SurrogateModel::fit(const Eigen::MatrixXd& predictor_set, const Eigen::MatrixXd& response_set, 
	std::string& surrogate_name) {
	if (surrogate_name == "GaussianProcess") {
		model_strategy_set = true;
		fitGaussianProcess(predictor_set, response_set);
	}
}

void SurrogateModel::fitGaussianProcess(const Eigen::MatrixXd& predictor_set, const Eigen::MatrixXd& response_set) {

}

Eigen::VectorXd SurrogateModel::predict(const Eigen::VectorXd& predictor) {
	GaussianProcess Agent;
	return Agent.predict(predictor);
}