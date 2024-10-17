#include "BayesianOptimization.h"
#include <random>
#include <NLopt/nlopt.hpp>

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

std::queue<Eigen::VectorXd*> BayesianOptimization::processInitializationData(const int& number_init) {
	/**�ڷ�Χ�ڽ������ѡȡpredictor
	* matlab��randomXFeasiblePoints��initialXFeasiblePoints���������������ɵ���������Ƿ���ϼȶ�Լ��
	*/

	std::queue<Eigen::VectorXd*> myQueue;
	std::mt19937 gen(42);

	/*��initialXFeasiblePoints�����л����������ɵ���ѵĳ�ʼ���������*/
	for (int i = 0; i < number_init; i++) {
		Eigen::VectorXd random_predictor(predictor_domainMin.size()); // Ҫ��new��
		for (int nd = 0; nd < predictor_domainMin.size(); nd++) {
			std::uniform_real_distribution<> dis(predictor_domainMin[nd], predictor_domainMax[nd]);
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
	init_try_point_queue = processInitializationData(number_init);

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

	/*matlabʹ��fminbndGlobalѰ��ȫ�����Ž⣬�����ҵ��ֲ�����*/

	return { candiate_predictor, candiate_response };
}

Eigen::VectorXd BayesianOptimization::findNextInAcquisitionFcn() {

	/*�����ʼ���õ�̽������δ���*/
	if (!init_try_point_queue.empty()) {
		Eigen::VectorXd* candiate_predictor = init_try_point_queue.front();
		init_try_point_queue.pop();
		return *candiate_predictor;
	}

	/*�����ʼ̽�������*/

	// matlab��shouldChooseRandomPoint����
	// �����Ǿ����Ƿ�Ӧ��ѡ��һ�������̽��Ŀ�꺯���ռ䣬�������������еĴ���ģ�ͽ���Ԥ����Ż���
	//
	// matlab��legalizePoints����
    // �����ǽ�ԭʼ���루����ת���ĵ㣩ӳ��ؿ��еĽ�ռ䣬��ȷ����Щ��������ⶨ�������Լ����
	//
	// matlab��ProbAllConstraintsSatisfied����
	// ������Լ���ĸ���

	/**ȫ�����Ż�
	* ���ڲɼ��������ҵ�����������ʹ�ò��������������ĵ㣬��Ϊ��һ��̽����
	*/
	Eigen::VectorXd candiate_predictor = fitAcquisitionFcn();

	return candiate_predictor;
}

std::vector<PointIncumbent> BayesianOptimization::findFBestGlobal(int& num_initial_points, int& num_topK_points) {
	std::vector<PointIncumbent> candiatesN; // �洢���еĺ�ѡ��
	std::vector<double> scores; // �洢��ѡ���Ŀ�꺯��ֵ

	/*����NLopt�����ȫ���Ż�*/
	nlopt::opt opt(nlopt::GN_CRS2_LM, predictor_domainMin.size());
	opt.set_lower_bounds(predictor_domainMin); // �����Ż��߽�
	opt.set_upper_bounds(predictor_domainMax);
	// 
	opt.set_max_objective([this](const Eigen::VectorXd& predictor) {
		return SurrogateModel::predict(predictor);
		}, nullptr);
	// ������ֹ�����������ݲ������������
	opt.set_xtol_rel(1e-6);
	opt.set_maxeval(100); // ����������

	std::mt19937 gen(42);
	// int num_initial_points = 1;
	// ��ÿ����ʼ��ִ���Ż�
	for (int i = 0; i < num_initial_points; i++) {
		std::vector<double> x0(predictor_dimension);

		// Ϊÿ��ά������һ�������ʼ�²�
		for (int nd = 0; nd < predictor_dimension; nd++) {
			std::uniform_real_distribution<> dis(predictor_domainMin[nd], predictor_domainMax[nd]);
			x0[nd] = dis(gen);
		}

		PointIncumbent candiate;
		double fbest;
		// ִ���Ż�
		nlopt::result result = opt.optimize(x0, fbest);
		// doubleת��ΪVectorXd
		candiate.candiate_predictor = Eigen::Map<Eigen::VectorXd>(x0.data(), x0.size()); 
		candiate.candiate_response = SurrogateModel::predict(candiate.candiate_predictor);
		candiatesN.emplace_back(candiate);
		scores.emplace_back(fbest); // ��ʼֵ��ʲô��
	}

	/*ѡ��ǰk����ѵ�*/
	std::vector<PointIncumbent> candicatesK;
	std::vector<int> index(candiatesN.size()); // ����candidatesN������
	for (int i = 0; i < candiatesN.size(); i++) {
		index[i] = i;
	}
	std::sort(index.begin(), index.end(), [&scores](int a, int b) {
		return scores[a] > scores[b]; 
		});
	for (int k = 0; k < num_topK_points; k++) {
		candicatesK.emplace_back(candiatesN[index[k]]);
	}

	return candicatesK;
}

