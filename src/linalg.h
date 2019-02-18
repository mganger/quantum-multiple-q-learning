#pragma once
#include <vector>
#include <ostream>
#include <iomanip>

using matrix = std::vector<std::vector<double>>;

matrix make_matrix(std::size_t y, std::size_t x, double init = 0){
	std::vector<double> v(x,init);
	matrix data(y,v);
	return data;
}

std::ostream& operator<<(std::ostream& o, matrix m){
	for(auto& row : m){
		for(auto& elem : row)
			o << std::fixed << std::setw(11) << std::setprecision(6) << elem;
		o << std::endl;
	}
	return o;
}

std::ostream& operator<<(std::ostream& o, std::vector<double> v){
	for(auto& e : v)
		o << std::fixed << std::setw(11) << std::setprecision(6) << e << std::endl;
	return o;
}

template<class T>
std::size_t argmax(T& a){
	std::size_t N = a.size(), mi = 0;
	for(std::size_t i = 1; i < N; i++)
		if(a[i] > a[mi])
			mi = i;
	return mi;
}

template <class T>
void norm1(T& vec){
	double sum = 0;
	for(auto& v : vec)
		sum += v;
	for(auto& v : vec)
		v /= sum;
}

template <class T>
void norm2(T& vec){
	double sum = 0;
	for(auto& v : vec)
		sum += v*v;
	sum = sqrt(sum);
	for(auto& v : vec)
		v /= sum;
}

template<class T>
double square_sum(T& vec){
	double s = 0;
	for(auto& v : vec)
		s += v*v;
	return s;
}

std::random_device rd;
std::mt19937 gen(rd());

auto random_choice(std::size_t a, std::size_t b){
	std::uniform_int_distribution<> dis(a,b);

	return dis(gen);
}

template<class T>
std::size_t weighted_choice(T& p){
	static std::random_device rd;
	static std::mt19937 gen(rd());
	static std::uniform_real_distribution<> dis(0,1);

	std::size_t N = p.size();
	double s = 0, r = dis(gen);
	for(std::size_t i = 0; i < N; i++)
		if((s+=p[i]) > r)
			return i;
	return N-1;
}

template<class T>
std::size_t square_weighted_choice(T& p){
	static std::random_device rd;
	static std::mt19937 gen(rd());
	static std::uniform_real_distribution<> dis(0,1);

	std::size_t N = p.size();
	double s = 0, r = dis(gen);
	for(std::size_t i = 0; i < N; i++)
		if((s+=p[i]*p[i]) > r)
			return i;
	return N-1;
}

template<class T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b){
	std::size_t n = std::min(a.size(), b.size());
	std::vector<T> c(n);
	for(std::size_t i = 0; i < n; i++)
		c[i] = a[i] + b[i];
	return c;
}

template <class T>
std::vector<T> operator/(std::vector<T> a, double b){
	for(auto& ai : a)
		ai = ai / b;
	return a;
}

std::vector<double> flatten(matrix m){
	std::vector<double> v;
	for(auto& row : m)
		for(auto& elem : row)
			v.push_back(elem);
	return v;
}
