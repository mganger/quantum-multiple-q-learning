#pragma once
#include "algorithm.h"
#include "linalg.h"
#include <ostream>

struct pqlearning : public algorithm {
	matrix<double> Q;
	matrix<double> p;

	pqlearning(Environment env, Parameters par_) :
		algorithm{env,par_},
		Q{make_matrix(env.Ns, env.Na, 0.0)},
		p{make_matrix(env.Ns, env.Na, 1.0/env.Na)}
	{}
	void episode() override {
		state_t s = env.s0;
		while(env.grid[s] != terminal){
			action_t a = static_cast<action_t>(weighted_choice(p[s]));
			state_t sp = env.next_state[s][a];
			double r   = env.reward[sp];
			double Vsp = Q[sp][argmax(Q[sp])];
			Q[s][a]   += par.alpha*(r + par.gamma*env.grid[sp]*Vsp - Q[s][a]);
			p[s][a]   += par.k*(r + Vsp);
			norm1(p[s]);
			s = sp;
		}
	}

	test_t test(std::size_t maxi) override {
		test_t t;
		state_t s = env.s0;
		for(;t.n_iter < maxi && env.grid[s] != terminal; t.n_iter++){
			s = env.next_state[s][weighted_choice(p[s])];
			t.reward += env.reward[s];
		}
		return t;
	}

	std::ostream& print(std::ostream& o) override {
		return o << "Q: " << std::endl << Q << std::endl << "p: " << std::endl << p;
	}
};
