#pragma once
#include "algorithm.h"
#include "linalg.h"
#include <ostream>

struct double_qlearning : public algorithm {
	matrix<double> Q1,Q2;

	double_qlearning(Environment& env_, Parameters par_) :
			algorithm{env_,par_},
			Q1{make_matrix(env.Ns,env.Na,0.0)},
			Q2{Q1}
	{}
	~double_qlearning(){}

	void episode() override {
		state_t s = env.s0;
		while(env.grid[s] != terminal){
			action_t a = static_cast<action_t>(random_choice(0,env.Na-1));
			state_t sp = env.next_state[s][a];
			double r   = env.reward[sp];
			auto choice = random_choice(0,1);
			if(choice == 0)
				Q1[s][a] += par.alpha*(r + par.gamma*env.grid[sp]*Q1[sp][argmax(Q2[sp])] - Q1[s][a]);
			else
				Q2[s][a] += par.alpha*(r + par.gamma*env.grid[sp]*Q2[sp][argmax(Q1[sp])] - Q2[s][a]);
			s = sp;
		}
	}

	test_t test(std::size_t maxi) override{
		test_t t;
		state_t s = env.s0;
		for(;t.n_iter < maxi && env.grid[s] != terminal; t.n_iter++){
			auto Q = Q1[s] + Q2[s];
			s = env.next_state[s][argmax(Q)];
			t.reward += env.reward[s];
			if(env.grid[s] == terminal)
				return t;
		}
		return t;
	}
	std::ostream& print(std::ostream& o) override {
		auto Q = (Q1+Q2)/2;
		auto N = Q.size();
		std::vector<double> v(N,0.0);
		for(std::size_t i = 0; i < N; i++)
			v[i] = Q[i][argmax(Q[i])];
		return o << "Q: " << std::endl << Q << std::endl << std::endl << "V (max Q):" << std::endl << v;
	}
};
