#include <fstream>
#include "TString.h"

void read_timestamp() {
    TString data_folder = "../source_data/angle_0_2_GeV/";
    TString write_file = "timestamp.txt";
    TString read_file = "hybrid0.root";

    TString write_path = data_folder + write_file;
    TString read_path = data_folder + read_file;

    ofstream myfile;
    myfile.open(write_path);

    TFile file(read_path);
    TTreeReader reader("hybrid;128", &file);
    TTreeReaderValue<float> timestamp(reader, "timestamp");

    while (reader.Next()) {
        myfile << *timestamp << endl;
    }
    myfile.close();
}
