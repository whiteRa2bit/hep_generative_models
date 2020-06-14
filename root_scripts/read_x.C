#include <fstream>
#include "TString.h"

void read_x() {
    TString data_folder = "../source_data/angle_0_100_GeV/";
    TString write_file = "x.txt";
    TString read_file = "hybrid0.root";

    TString write_path = data_folder + write_file;
    TString read_path = data_folder + read_file;

    ofstream myfile;
    myfile.open(write_path);

    TFile file(read_path);
    TTreeReader reader("hybrid;41", &file);
    TTreeReaderValue<float> x(reader, "x");

    while (reader.Next()) {
        myfile << *x << endl;
    }
    myfile.close();
}
