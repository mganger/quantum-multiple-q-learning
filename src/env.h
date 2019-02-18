#pragma once

#include <array>
#include <vector>
#include <random>
#include <string>
#include <istream>
#include <iostream>
#include "linalg.h"

typedef std::size_t state_t;
typedef int64_t dim_t;

enum action_t {
	up=0,
	down=1,
	left=2,
	right=3,
};

enum rule {
	adjacent=4,
	diagonal=8,
	duplicate=16,
};

enum position_t {
	terminal=0, normal, invalid
};

struct Environment {
	using size_t = std::size_t;

	std::vector<position_t> grid;
	std::vector<double> reward_;
	std::vector<std::vector<state_t>> next_state;
	const size_t X,Y,Ns,Na;
	const state_t s0;
	double rs;

	template<class Function>
	Environment(dim_t x0, dim_t y0, rule tr, Function reward_func, double rs_, std::vector<std::string> grid_) :
				X{grid_[0].size()}, Y{grid_.size()}, Ns{X*Y}, Na{tr}, s0{y0*X+x0}, rs{rs_}{

		for(size_t i = 0; i < Y; i++)
			for(size_t j = 0; j < X; j++)
				grid.push_back(grid_[i][j] == '-' ? normal : grid_[i][j] == '+' ? terminal : invalid),
				reward_.push_back(reward_func(i,j));

		for(state_t s = 0; s < Ns; s++)
			next_state.push_back(next_states(s,tr));
	}

	double reward(state_t s){
		return reward_[s] + (2*rs*random_choice(0,1)-rs)*grid[s];
	}

	std::vector<state_t> next_states(state_t s, rule r){
		dim_t x = s % X;
		dim_t y = s / X;

		auto choose = [&](bool a, auto sp){return a && (grid[sp] != invalid) ? sp : s;};

		switch(r){
			case diagonal: return {
				choose(y > 0    , s - X),
				choose(y < Y - 1, s + X),
				choose(x > 0    , s - 1),
				choose(x < X - 1, s + 1),
				choose(y > 0     && x > 0,     s - X - 1),
				choose(y > 0     && x < X - 1, s - X + 1),
				choose(y < Y - 1 && x > 0,     s + X - 1),
				choose(y < Y - 1 && x < X - 1, s + X + 1),
			};
			case duplicate: {
				std::vector<state_t> v;
				for(std::size_t i = 0; i < duplicate/4; i++)
					v.push_back(choose(y > 0    , s - X)),
					v.push_back(choose(y < Y - 1, s + X)),
					v.push_back(choose(x > 0    , s - 1)),
					v.push_back(choose(x < X - 1, s + 1));
				return v;
			}
			case adjacent:
			default: return {
				choose(y > 0    , s - X),
				choose(y < Y - 1, s + X),
				choose(x > 0    , s - 1),
				choose(x < X - 1, s + 1),
			};
		}

	}

};

Environment make_environment(std::istream& file){
	std::vector<std::string> grid;
	std::string line;
	while(std::getline(file, line) && !line.empty())
		grid.push_back(line);

	std::size_t Y = grid.size(), X = grid[0].size();

	std::string type;
	file >> type;
	rule tr =
		type == "diagonal"  ? diagonal  :
		type == "duplicate" ? duplicate :
		                      adjacent;

	dim_t x0, y0;
	double r0,s;
	file >> y0 >> x0 >> r0 >> s;

	std::vector<double> reward_map(X*Y,r0);
	dim_t x,y;
	double r;
	while(file >> x >> y >> r){
		reward_map[y*X+x] = r;
	}
	std::cerr << "Grid Size: " << X << "x" << Y << std::endl;
	std::cerr << "Reward:    " << r0 << " +- " << s << std:: endl;
	std::cerr << "Initial:   (" << x0 << "," << y0 << ")" << std::endl;
	std::cerr << "Goal:      (" << x  << "," << y  << ")" << std::endl;
	return {x0,y0,tr,[=](dim_t x, dim_t y){return reward_map[y*X+x];},s,grid};
}
