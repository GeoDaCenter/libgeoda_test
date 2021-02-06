//
// Created by Xun Li on 2/5/21.
//

#include <iostream>
#include <vector>
#include <chrono>
using namespace std::chrono;

#include <libgeoda/libgeoda.h>
#include <libgeoda/weights/GeodaWeight.h>
#include <libgeoda/sa/LISA.h>
#include <libgeoda/sa/BatchLISA.h>
#include <libgeoda/gda_weights.h>
#include <libgeoda/gda_sa.h>


int main()
{
    GeoDa gda("../../data/natregimes.shp");
    GeoDaWeight* w = gda_queen_weights(&gda, 1, false, 0);
    std::vector<double> hr = gda.GetNumericCol("HR60");
    std::vector<bool> undefs;

    LISA* lisa = gda_localmoran(w, hr, undefs, 0.01, 8, 99999, 123456789);

    std::vector<int> cvals= lisa->GetClusterIndicators();
    std::vector<double> pvals = lisa->GetLocalSignificanceValues();
    std::vector<double> mvals = lisa->GetLISAValues();
    delete lisa;
    delete w;
}