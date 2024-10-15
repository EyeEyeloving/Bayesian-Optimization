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

std::queue<Eigen::VectorXd*> BayesianOptimization::processInitializationData(const Eigen::VectorXd& x_lower, const Eigen::VectorXd& x_upper, const int& number_init) {
	/**�ڷ�Χ�ڽ������ѡȡpredictor
	* matlab��randomXFeasiblePoints��initialXFeasiblePoints���������������ɵ���������Ƿ���ϼȶ�Լ��
	*/

	std::queue<Eigen::VectorXd*> myQueue;
	std::mt19937 gen(42);

	/*��initialXFeasiblePoints�����л����������ɵ���ѵĳ�ʼ���������*/
	for (int i = 0; i < number_init; i++) {
		Eigen::VectorXd random_predictor(x_lower.rows()); // Ҫ��new��
		for (int nd = 0; nd < x_lower.rows(); nd++) {
			std::uniform_real_distribution<> dis(x_lower(nd), x_upper(nd));
			random_predictor(nd) = dis(gen);
		}
		myQueue.emplace(&random_predictor);
	}

	/*�������У��鿴��ʼ��������û������*/
	for (int i = 0; i < number_init; i++) {
		Eigen::VectorXd* x = myQueue.front();
		myQueue.pop();
		std::cout << x->transpose() << std::endl;
		myQueue.emplace(x);
	}

	return myQueue;
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

	/**��ʼ����Ҫ����׼���Ĺ۲�������
	* �������һЩpredictor��Ȼ����response����������{predictor��response}
	* ������Ԥ����������ڣ������Ƕ���ʵ����Ĳ�������ִ��һ����ʵ��trial���õ����������Ԥ��ֻ��ͨ��ģ�ͻ�������������Ԥ��
	* ����׼����/��Ҫ���в�����predictorͨ�����д洢
	*/
	Eigen::VectorXd x_lower = data_block.colwise().minCoeff(); // ��������д
	Eigen::VectorXd x_upper = data_block.colwise().maxCoeff();
	int number_init = 5;
	init_try_point_queue = processInitializationData(x_lower, x_upper, number_init);

	// Eigen::VectorXd candiate_predictor(predictor_dimension);
	// candiate_predictor.setZero();

	/*��ʼ������ģ��*/
	fitSurrogateModel();

	/**�������ź�ѡ�㣬��Ӧmatlab����findIncumbent
	* �ҵ�ʹ����ģ�ͣ����磬GP�����ţ����磬����ֵλ�ã���predictorλ�ü���Ԥ��response
	*/
	auto candiate = findIncumbent();

	//Eigen::VectorXd candiate_predictor;
	//Eigen::VectorXd candiate_response;

	/*���ڲ�������ѡ����һ��predictor*/
	Eigen::VectorXd candiate_predictor = findNextInAcquisitionFcn();
	
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

		/*������һ�����ź�ѡ��*/
		candiate = findIncumbent();

		/*���ڲ�������ѡ����һ��predictor*/
		candiate_predictor = findNextInAcquisitionFcn();

		/*��Ҷ˹�Ż�����+1*/
		++number_bayesopt;

		/*�����˳�������������*/
		if (number_bayesopt == number_max_iter) optimization_finished = true;
	}
}

void BayesianOptimization::fitSurrogateModel() {
	SurrogateModel::fit(predictor_set, response_set, surrogate_model);
}

Eigen::VectorXd BayesianOptimization::fitAcquisitionFcn() {
	AcquisitionFcn::fit(acquisition_func);
	return AcquisitionFcn::getNextCandiatePoint();
}

PointIncumbent BayesianOptimization::findIncumbent() {
	Eigen::VectorXd candiate_predictor(predictor_dimension);
	Eigen::VectorXd candiate_response(response_dimension);

	// matlabʹ��fminbndGlobalѰ��ȫ�����Ž�

}

Eigen::VectorXd BayesianOptimization::findNextInAcquisitionFcn() {

	/*�����ʼ���õ�̽������δ���*/
	if (!init_try_point_queue.empty()) {
		Eigen::VectorXd* candiate_predictor = init_try_point_queue.front();
		init_try_point_queue.pop();
		return *candiate_predictor;
	}

	/*�����ʼ̽�������*/
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

