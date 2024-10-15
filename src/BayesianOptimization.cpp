#include "BayesianOptimization.h"
#include <random>

BayesianOptimization::BayesianOptimization()
	: surrogate_model("GaussianProcess"), acquisition_func("ExpectedImprovement") {

}

BayesianOptimization::BayesianOptimization(std::string surrogate, std::string acquisition)
	: surrogate_model(surrogate), acquisition_func(acquisition) {

}

Eigen::MatrixXd BayesianOptimization::validateDataInput(Eigen::MatrixXd& data_block, int& data_dimension) {
	predictor_dimension = data_dimension;
	if (data_block.rows() == predictor_dimension) return data_block;
	return data_block.transpose();
}

Eigen::VectorXd BayesianOptimization::callObjectiveFcn(const Eigen::VectorXd& candiate_predictor) {
	/*����ʵ�����в��������������Ŀ�꺯��ֵ*/
	Eigen::VectorXd evaluation_response(response_dimension);
	return evaluation_response.setZero();
}

void BayesianOptimization::augmentObservationSet(const Eigen::VectorXd candiate_predictor, const Eigen::VectorXd candiate_response) {
	// ������Ҫ�ÿ�������������ý������ⲿ�޸�candiate_predictorֵҲ����set��ֵ�����޸�
	predictor_set << candiate_predictor;
	response_set << candiate_response;
}

void BayesianOptimization::updateTrace() {

}

void BayesianOptimization::fit(Eigen::MatrixXd& data_block_raw, int& data_dimension) {
	
	/*�����������*/

	Eigen::MatrixXd data_block = validateDataInput(data_block_raw, data_dimension);
	
	/*��ʼ������ģ��*/
	//Eigen::VectorXd candiate_predictor = findNextInAcquisitionFcn();
	Eigen::VectorXd candiate_predictor(predictor_dimension);
	candiate_predictor.setZero();
	
	/*���뱴Ҷ˹�Ż�*/

	while (!optimization_finished) {
		/*��Ŀ�꺯�����в���*/
		Eigen::VectorXd evaluation_resopnse = callObjectiveFcn(candiate_predictor);
		/*������֪���ݾ���*/
		augmentObservationSet(candiate_predictor, evaluation_resopnse);
		/*����ѵ������ģ��*/
		fitSurrogateModel();
		/*��¼��Ҷ˹�Ż�����*/
		updateTrace();

		/*ѡ����һ����ѡ��*/
		candiate_predictor = findNextInAcquisitionFcn();
		++number_bayesopt;
	}
}

void BayesianOptimization::fitSurrogateModel() {
	SurrogateModel::fit(predictor_set, response_set, surrogate_model);
}

Eigen::VectorXd BayesianOptimization::fitAcquisitionFcn() {
	AcquisitionFcn::fit(acquisition_func);
	return AcquisitionFcn::getNextCandiatePoint();
}

Eigen::VectorXd BayesianOptimization::findNextInAcquisitionFcn() {
	// ���Incumbent��Ŀ�ģ�

	// ���û�к�ѡ��std::queue��ҪcallObjectivefcn��



}

/*matlab��run�����е�findIncumbent���������ܲ�����Ҫ���ֽ׶���ɽ��*/
//NextPointCandiate BayesianOptimization::findNextPointCandiate(const Eigen::VectorXd& x_lower, const Eigen::VectorXd& x_upper) {
//	Eigen::VectorXd candiate_predictor(predictor_dimension);
//	Eigen::VectorXd candiate_response(response_dimension);
//
//	/*Ϊ���ҵ������ʵ���һ�������㣬matlabʹ���˺���fminbndGlobal��fminsearch*/ 
//
//	// ���ɸ�����Χ�ڵ������
//	// int number_candiate = 1;
//	std::mt19937 gen(42);
//	for (int nd = 0; nd < predictor_dimension; nd++) {
//		std::uniform_real_distribution<> dis(x_lower(nd), x_upper(nd));
//		candiate_predictor(nd) = dis(gen);
//	}
//	// ���������ĺ���ֵ
//	candiate_response.setConstant(1);
//	
//	return {candiate_predictor, candiate_response};
//}

