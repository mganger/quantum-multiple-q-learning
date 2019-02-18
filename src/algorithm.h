#pragma once
#include "env.h"
#include "path.h"
#include <fstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>

#define MAIN(algo) int main(int argc, char**argv){ return main_func<algo>(argc,argv); }

struct test_t {
	std::size_t n_iter = 0;
	double reward = 0;
	std::vector<double> double_output;
	path_t path;
};

test_t average(std::vector<test_t>& v){
	test_t t = v[0];
	auto n = v.size();
	for(std::size_t i = 1; i < n; i++){
		t.n_iter += v[i].n_iter;
		t.reward += v[i].reward;
		auto ni = v[i].double_output.size();
		for(std::size_t j = 0; j < ni; j++)
			t.double_output[j] += v[i].double_output[j];
	}
	t.reward /= n;
	for(auto& x : t.double_output)
		x /= n;
	return t;
}

std::ostream& operator<<(std::ostream& o, test_t t){
	auto w = std::setw(o.width());
	o << t.n_iter << " " << w << t.reward;
	for(auto& x : t.double_output)
		o << " " << w << x;
	return o;
}

struct Parameters {
	double alpha, gamma, k, epsilon;
	std::size_t n_episodes;
};

Parameters make_parameters(std::ifstream& file){
	Parameters p;
	std::string param;
	while(file >> param && !param.empty()){
		if(param == "alpha")
			file >> p.alpha;
		else if(param == "gamma")
			file >> p.gamma;
		else if(param == "k")
			file >> p.k;
		else if(param == "epsilon")
			file >> p.epsilon;
		else if(param == "episodes")
			file >> p.n_episodes;
	}
	std::cerr << "Alpha:    " << p.alpha      << std::endl;
	std::cerr << "Gamma:    " << p.gamma      << std::endl;
	std::cerr << "Epsilon:  " << p.epsilon    << std::endl;
	std::cerr << "Episodes: " << p.n_episodes << std::endl;
	std::cerr << "k:        " << p.k          << std::endl;
	return p;
}

struct algorithm {
	Environment env;
	Parameters par;
	algorithm(Environment env_, Parameters par_) : env{env_}, par{par_} {}
	~algorithm(){};
	virtual void episode()=0;
	virtual test_t test(std::size_t)=0;
	virtual std::ostream& print(std::ostream& o)=0;

};

template<class T>
int main_func(int argc, char** argv){
	if(argc <= 2){
		std::cout << "<parameter_file> <environment_file> [<n_iter> <n_steps_convergence>]" << std::endl;
		return 1;
	}

	std::ifstream parameter_file{argv[1]}, env_file{argv[2]};

	if(!parameter_file.good()){
		std::cout << "Invalid file " << argv[1] << std::endl;
		return 2;
	}
	if(!env_file.good()){
		std::cout << "Invalid file " << argv[2] << std::endl;
		return 3;
	}

	std::size_t n_runs = 1;
	if(argc > 3){
		n_runs = std::stoul(argv[3]);
		n_runs = n_runs > 0 ? n_runs : 1;
	}

	Parameters p = make_parameters(parameter_file);
	Environment env = make_environment(env_file);

	//if test_convergence is defined, run each algorithm individually compute the number of steps until convergence
	if(argc >= 5){
		auto n_steps = std::stoul(argv[4]);


		std::mutex mtx;
		std::atomic<std::size_t> n{0}, s{0}, ss{0}, time_ns{0};
		std::vector<std::thread> workers;
		for(std::size_t j = 0; j < std::thread::hardware_concurrency(); j++)
			workers.emplace_back([&]{
				while(true){
					mtx.lock();
					if(n >= n_runs){
						mtx.unlock();
						break;
					}
					n++;
					mtx.unlock();
					T algo{env,p};
					std::size_t i = 0;
					for(; i < 10000; i++){
						auto start = std::chrono::high_resolution_clock::now();
						algo.episode();
						auto finish = std::chrono::high_resolution_clock::now();
						auto result = algo.test(1000);
						time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
						if(result.n_iter == n_steps)
							break;
					}
					s  += i;
					ss += i*i;
				}
			}
		);
		for(auto& worker : workers)
			worker.join();

		std::cout << double(s)/n << " " << sqrt(double(ss)/n) << " " << (sqrt(double(ss))/n) << " " << time_ns << " " << s << " " << n << std::endl;
	}else{
		std::vector<T> algos(n_runs,T(env,p));
		std::cout << std::left;
		std::vector<std::vector<test_t>> tss;
		std::mutex mtx;

		for(int ep = 0; ep < p.n_episodes; ep++){
			std::vector<std::thread> workers;
			std::vector<test_t> ts;
			auto n_thread = std::min(std::size_t(std::thread::hardware_concurrency()), algos.size());
			for(std::size_t j = 0; j < n_thread; j++)
				workers.emplace_back([&,j,n_thread]{
					auto k     = algos.size()/n_thread*j;
					auto end   = j != n_thread - 1 ? algos.size()/n_thread*(j+1) : algos.size();
					std::vector<test_t> ts_thread;
					for(; k < end; k++){
						algos[k].episode();
						ts_thread.push_back(algos[k].test(10000));
					}
					mtx.lock();
					ts.insert(ts.end(),ts_thread.begin(),ts_thread.end());
					mtx.unlock();
				});
			for(auto& worker : workers)
				worker.join();
			auto branching = group(ts,10);
			for(auto& branch : branching)
				std::cout << std::setw(6) << ep << std::setw(10) << branch.first << " " << std::setw(10) << branch.second.size() << " " << std::setw(8) << average(branch.second) << std::endl;
		}
	}
	return 0;
}

std::ostream& operator<<(std::ostream& o, algorithm& a){
	return a.print(o);
}
