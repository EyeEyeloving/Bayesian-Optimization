#include "SurrogateModel.h"

SurrogateModel::SurrogateModel()
	: surrogate_model("GaussianProcess") {

}

SurrogateModel::SurrogateModel(std::string surrogate)
	: surrogate_model(surrogate) {

}

void SurrogateModel::fit() {

}

void SurrogateModel::fitGaussianProcess() {

}