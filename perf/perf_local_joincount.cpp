//
// Created by Xun Li on 2/5/21.
//

#include <iostream>
#include <vector>
#include <libgeoda/libgeoda.h>
#include <libgeoda/weights/GeodaWeight.h>
#include <libgeoda/sa/LISA.h>
#include <libgeoda/gda_weights.h>
#include <libgeoda/gda_sa.h>


int main()
{
    GeoDa gda("../../data/deaths_nd_by_house.shp");
    GeoDaWeight* w = gda_distance_weights(&gda, 20, "", 1.0, false, false, false, "", false);
    std::vector<double> data = gda.GetNumericCol("death_dum");

    int nCPUs = 6;
    int permutations = 99999;
    int last_seed_used = 123456789;
    double significance_cutoff = 0.01;

    LISA* jc = gda_localjoincount(w, data, std::vector<bool>(),significance_cutoff, nCPUs, permutations, last_seed_used);

    std::vector<int> nnvals = jc->GetNumNeighbors();
    std::vector<double> pvals = jc->GetLocalSignificanceValues();
    std::vector<double> jvals = jc->GetLISAValues();

    std::cout << pvals[1] << std::endl;

    delete jc;
    delete w;
}