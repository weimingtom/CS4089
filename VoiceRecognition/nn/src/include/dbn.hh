/*
 * $File: dbn.hh
 */

#pragma once

#include "rbm.hh"

/// Deep Belief Network, Layers of RBMs
class DBN {
	public:
		std::vector<RBM *> rbms;

		// DO NOT take the possession of rbm given
		void add_rbm(RBM *rbm);

		void fit_last_layer(std::vector<std::vector<real_t>> &X);
};

/**
 * vim: syntax=cpp11 foldmethod=marker
 */
