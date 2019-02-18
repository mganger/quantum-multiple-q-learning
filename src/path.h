#pragma once
#include "env.h"
#include <algorithm>
#include <map>
#include <ostream>
#include <sstream>

typedef std::vector<action_t> path_t;

struct path_cmp {
	bool operator()(const path_t& a, const path_t& b){
		return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
	}
};

std::ostream& operator<<(std::ostream& o, const path_t& p){
	std::stringstream s;
	for(auto a : p)
		s << a;
	o << s.str();
	return o;
}

template<template<class> class V, class T>
auto filter(V<T> paths, path_t branch){
	V<T> output;
	auto pred = [&branch](auto v){return std::equal(branch.begin(), branch.end(), v.path);};
	std::copy_if(paths.begin(), paths.end(), std::back_inserter(output), pred);
	return output;
}

template<class V>
auto group(V paths, std::size_t maxp = 10){
	std::map<path_t, V,path_cmp> m;
	for(auto& p : paths)
		for(auto iter = p.path.begin(); iter < p.path.end() && iter < p.path.begin() + maxp; ++iter){
			path_t key(p.path.begin(), iter+1);
			m[key].push_back(p);
		}
	return m;
}
