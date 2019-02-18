#include "algorithm.h"
#include "linalg.h"
#include <ostream>

template<std::size_t N>
struct multi_qlearning : public algorithm {
	std::vector<matrix> Q;

	multi_qlearning(Environment& env_, Parameters par_) :
			algorithm{env_,par_},
			Q(N,make_matrix(env.Ns,env.Na,0.0))
	{}
	~multi_qlearning(){}

	void episode() override {
		state_t s = env.s0;
		while(env.grid[s] != terminal){
			action_t a = static_cast<action_t>(random_choice(0,env.Na-1));
			state_t sp = env.next_state[s][a];
			double r   = env.reward(sp);
			auto choice = random_choice(0,N-1);
			auto ap = argmax(Q[choice][sp]);
			double Vsp = -Q[choice][sp][ap];
			for(std::size_t i = 0; i < N; i++)
				Vsp += Q[i][sp][ap];
			Vsp /= N;
			Q[choice][s][a] += par.alpha*(r + par.gamma*env.grid[sp]*Vsp - Q[choice][s][a]);
			s = sp;
		}
	}

	test_t test(std::size_t maxi) override{
		test_t t;
		state_t s = env.s0;
		auto Qavg = Q[0];
		for(std::size_t i = 1; i < N; i++)
			Qavg = Qavg + Q[i];

		for(;t.n_iter < maxi && env.grid[s] != terminal; t.n_iter++){
			auto a = static_cast<action_t>(argmax(Qavg[s]));
			t.path.push_back(a);
			s = env.next_state[s][a];
			t.reward += env.reward(s);
		}
		t.double_output = flatten(Qavg/N);
		return t;
	}
	std::ostream& print(std::ostream& o) override {
		auto Qavg = Q[0];
		for(std::size_t i = 1; i < N; i++)
			Qavg = Qavg + Q[i];
		Qavg = Qavg/N;
		auto Ns = Qavg.size();
		std::vector<double> v(Ns,0.0);
		for(std::size_t i = 0; i < Ns; i++)
			v[i] = Qavg[i][argmax(Qavg[i])];
		return o << "Q: " << std::endl << Qavg << std::endl << std::endl << "V (max Q):" << std::endl << v;
	}
};
