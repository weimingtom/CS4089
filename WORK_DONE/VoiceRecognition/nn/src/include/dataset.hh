/*
 * $File: dataset.hh
 */

#pragma once

#include "type.hh"

#include <vector>

typedef std::vector<std::pair<int, real_t> >	Instance;
typedef std::vector<Instance>					Dataset;
typedef std::vector<Instance *>					RefDataset;
typedef std::vector<const Instance *>			ConstRefDataset;
typedef std::vector<int>						Labels;
typedef std::vector<real_t>						RealLabels;

typedef std::vector<real_t>						Vector;

/**
 * vim: syntax=cpp11 foldmethod=marker
 */
